#include "config.h"

#include "oboe.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

#include "oboe/Oboe.h"

#if HAVE_CXXMODULES
import alstring;
import backends.exception;
import core.device;
import gsl;
import logging;
import numeric;
import ringbuffer;
#else
#include "alc/backends/exception.hpp"
#include "alstring.h"
#include "core/device.h"
#include "core/logging.h"
#include "gsl/gsl"
#include "numeric.hpp"
#include "ringbuffer.hpp"
#endif


namespace {

    using namespace std::string_view_literals;

    [[nodiscard]] constexpr auto GetDeviceName() noexcept { return "Oboe Default"sv; }


    /* Oboe playback. The stream can die at any time: a phone call, the output device changing, the
     * audio server restarting, a long sleep. A playing stream is then reported through the error
     * callback, but a stopped one (the device paused while the app is in the background) is not: AAudio
     * only marks it disconnected, and it can't be started again. Either way the stream is opened again,
     * with the format the device already mixes in (OpenAL's mixer is set up for it), and started again
     * if OpenAL wants it playing. start() never gives up on a dead stream by throwing: OpenAL would then
     * disconnect the device for good (no sound until the app reopens it), it keeps trying instead.
     */
    struct OboePlayback final : BackendBase, oboe::AudioStreamCallback {
        using BackendBase::BackendBase;
        ~OboePlayback() override;

        /* All under mStreamMutex, except the callback counter */
        std::shared_ptr<oboe::AudioStream> mStream;
        std::mutex mStreamMutex;
        std::condition_variable mWatchCv;
        std::thread mWatchThread;   /* Reopens a dead stream, watches for a stalled one */
        bool mPlaying{false};       /* OpenAL started the device (and did not stop it since) */
        bool mReopenPending{false}; /* The stream died while playing, the watch thread opens it again */
        bool mQuit{false};
        std::atomic<uint64_t> mCallbackCount{0u};

        auto onAudioReady(oboe::AudioStream *oboeStream, void *audioData, int32_t numFrames)
        -> oboe::DataCallbackResult override;

        void onErrorAfterClose(oboe::AudioStream* audioStream, oboe::Result error) override;

        void open(std::string_view name) override;
        auto reset() -> bool override;
        void start() override;
        void stop() override;

        void closeStream();
        auto openStreamLikeDevice() -> oboe::Result;
        auto startStream() -> oboe::Result;
        auto reopenStream() -> oboe::Result;
        void requestReopen();
        void watchThreadProc();
    };

    /* How often the watch thread checks a playing stream, and after how many checks without a data
     * callback it is regarded as stalled (some drivers stop calling back after a long sleep, with no
     * error).
     */
    constexpr auto WatchInterval = std::chrono::milliseconds{1000};
    constexpr auto StalledChecks = 3u;

    /* A stream that can't be opened again right away (the audio system may not be ready yet, e.g.
     * right after a call) is tried again for as long as OpenAL wants it playing, waiting a little
     * longer each time (up to 8 times this).
     */
    constexpr auto ReopenRetryDelay = std::chrono::milliseconds{250};

    OboePlayback::~OboePlayback()
    {
        {
            std::lock_guard<std::mutex> lock{mStreamMutex};
            mQuit = true;
        }
        mWatchCv.notify_all();
        if(mWatchThread.joinable())
            mWatchThread.join();

        std::lock_guard<std::mutex> lock{mStreamMutex};
        closeStream();
    }


    auto OboePlayback::onAudioReady(oboe::AudioStream *const oboeStream, void *const audioData,
                                    int32_t const numFrames) -> oboe::DataCallbackResult
    {
        mDevice.renderSamples(audioData, gsl::narrow_cast<uint32_t>(numFrames),
                               gsl::narrow_cast<uint32_t>(oboeStream->getChannelCount()));
        mCallbackCount.fetch_add(1u, std::memory_order_relaxed);
        return oboe::DataCallbackResult::Continue;
    }

    void OboePlayback::onErrorAfterClose(oboe::AudioStream *const audioStream, oboe::Result const error)
    {
        std::lock_guard<std::mutex> lock{mStreamMutex};

        /* Only the current stream matters: one closed or replaced meanwhile was dealt with already */
        if(mQuit || !mStream || mStream.get() != audioStream)
            return;

        /* Any error: Oboe stopped and closed it, it's dead */
        WARN("Oboe stream closed after error {}, opening it again", oboe::convertToText(error));
        mStream.reset();

        /* Not playing: start() opens it again when OpenAL wants it playing */
        if(mPlaying)
            requestReopen();
    }

    /* Asks the watch thread to open the stream again (mStreamMutex held) */
    void OboePlayback::requestReopen()
    {
        mReopenPending = true;
        if(!mWatchThread.joinable())
            mWatchThread = std::thread{&OboePlayback::watchThreadProc, this};
        mWatchCv.notify_all();
    }

    void OboePlayback::watchThreadProc()
    {
        std::unique_lock<std::mutex> lock{mStreamMutex};
        auto reopenAttempts = 0u;
        auto stalledChecks = 0u;
        auto lastCallbackCount = mCallbackCount.load(std::memory_order_relaxed);

        while(!mQuit)
        {
            if(mReopenPending)
            {
                /* Stopped, or opened by start() meanwhile */
                if(!mPlaying || mStream)
                {
                    mReopenPending = false;
                    reopenAttempts = 0u;
                    continue;
                }

                if(const auto result = reopenStream(); result == oboe::Result::OK)
                {
                    WARN("Oboe stream opened again");
                    mReopenPending = false;
                    reopenAttempts = 0u;
                    stalledChecks = 0u;
                    lastCallbackCount = mCallbackCount.load(std::memory_order_relaxed);
                    continue;
                }
                else if(reopenAttempts == 0u)
                {
                    ERR("Failed to open the Oboe stream again: {}, trying again",
                        oboe::convertToText(result));
                }

                ++reopenAttempts;
                mWatchCv.wait_for(lock, ReopenRetryDelay * std::min(reopenAttempts, 8u),
                                  [this]{ return mQuit; });
                continue;
            }

            mWatchCv.wait_for(lock, WatchInterval, [this]{ return mQuit || mReopenPending; });
            if(mQuit || mReopenPending)
                continue;

            /* A playing stream that stopped calling back: dead without telling */
            const auto callbackCount = mCallbackCount.load(std::memory_order_relaxed);
            const auto state = (mStream) ? mStream->getState() : oboe::StreamState::Closed;

            if(mPlaying && (callbackCount == lastCallbackCount)
                && (state == oboe::StreamState::Started || state == oboe::StreamState::Disconnected))
            {
                if(state == oboe::StreamState::Disconnected || ++stalledChecks >= StalledChecks)
                {
                    WARN("Oboe stream {} (no data callbacks), opening it again",
                         (state == oboe::StreamState::Disconnected) ? "disconnected" : "stalled");
                    closeStream();
                    stalledChecks = 0u;
                    mReopenPending = true;
                }
            }
            else
            {
                stalledChecks = 0u;
            }

            lastCallbackCount = callbackCount;
        }
    }

    /* Closes the stream if there is one (mStreamMutex held) */
    void OboePlayback::closeStream()
    {
        if(mStream)
        {
            mStream->stop();
            mStream->close();
            mStream.reset();
        }
    }

    /* Opens the stream again with the format the device already uses (mStreamMutex held): the sample
     * rate, channel count and sample type OpenAL mixes in, Oboe converting to what the output wants now.
     */
    auto OboePlayback::openStreamLikeDevice() -> oboe::Result
    {
        auto format = oboe::AudioFormat::Unspecified;
        switch(mDevice.FmtType)
        {
            case DevFmtShort:
                format = oboe::AudioFormat::I16;
                break;
            case DevFmtFloat:
                format = oboe::AudioFormat::Float;
                break;
#if OBOE_VERSION_MAJOR > 1 || (OBOE_VERSION_MAJOR == 1 && OBOE_VERSION_MINOR >= 6)
            case DevFmtInt:
                format = oboe::AudioFormat::I32;
                break;
#endif
            default:
                ERR("Can't open the Oboe stream again for {} samples", DevFmtTypeString(mDevice.FmtType));
                return oboe::Result::ErrorInvalidFormat;
        }

        const auto channels = gsl::narrow_cast<int32_t>(mDevice.channelsFromFmt());
        const auto sampleRate = gsl::narrow_cast<int32_t>(mDevice.mSampleRate);

        auto builder = oboe::AudioStreamBuilder{};
        builder.setDirection(oboe::Direction::Output);
        builder.setPerformanceMode(oboe::PerformanceMode::LowLatency);
        builder.setUsage(oboe::Usage::Game);
        builder.setSharingMode(oboe::SharingMode::Shared);
        builder.setSampleRate(sampleRate);
        builder.setChannelCount(channels);
        builder.setFormat(format);
        builder.setSampleRateConversionQuality(oboe::SampleRateConversionQuality::Medium);
        builder.setChannelConversionAllowed(true);
        builder.setFormatConversionAllowed(true);
        builder.setCallback(this);

        if(const auto result = builder.openStream(mStream); result != oboe::Result::OK)
        {
            mStream.reset();
            return result;
        }

        if(mStream->getSampleRate() != sampleRate || mStream->getChannelCount() != channels
            || mStream->getFormat() != format)
        {
            ERR("Oboe stream opened again with another format:\n{}", oboe::convertToText(mStream.get()));
            closeStream();
            return oboe::Result::ErrorInvalidFormat;
        }

        mStream->setBufferSizeInFrames(std::min(gsl::narrow_cast<int32_t>(mDevice.mBufferSize),
                                                mStream->getBufferCapacityInFrames()));
        TRACE("Got stream with properties:\n{}", oboe::convertToText(mStream.get()));
        return oboe::Result::OK;
    }

    /* Starts the stream (mStreamMutex held): fails for a dead one */
    auto OboePlayback::startStream() -> oboe::Result
    {
        if(!mStream)
            return oboe::Result::ErrorClosed;

        switch(mStream->getState())
        {
            case oboe::StreamState::Starting:
            case oboe::StreamState::Started:
                /* AAudio refuses to start a started stream */
                return oboe::Result::OK;
            case oboe::StreamState::Disconnected:
            case oboe::StreamState::Closing:
            case oboe::StreamState::Closed:
                return oboe::Result::ErrorDisconnected;
            default:
                break;
        }

        return mStream->start();
    }

    /* Closes the stream and opens it again, started if OpenAL wants it playing (mStreamMutex held) */
    auto OboePlayback::reopenStream() -> oboe::Result
    {
        closeStream();

        auto result = openStreamLikeDevice();
        if(result == oboe::Result::OK && mPlaying)
        {
            result = mStream->start();
            if(result != oboe::Result::OK)
                closeStream();
        }
        return result;
    }

    void OboePlayback::open(std::string_view name)
    {
        if(name.empty())
            name = GetDeviceName();
        else if(name != GetDeviceName())
            throw al::backend_exception{al::backend_error::NoDevice, "Device name \"{}\" not found",
                                        name};

        /* Open a basic output stream, just to ensure it can work. */
        auto stream = std::shared_ptr<oboe::AudioStream>{};
        const auto result = oboe::AudioStreamBuilder{}.setDirection(oboe::Direction::Output)
                ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
                ->openStream(stream);
        if(result != oboe::Result::OK)
            throw al::backend_exception{al::backend_error::DeviceError, "Failed to create stream: {}",
                                        oboe::convertToText(result)};

        stream->close();
        mDeviceName = name;
    }

    auto OboePlayback::reset() -> bool
    {
        std::lock_guard<std::mutex> lock{mStreamMutex};

        /* OpenAL stops the device before resetting it */
        mPlaying = false;
        mReopenPending = false;
        closeStream();

        auto builder = oboe::AudioStreamBuilder{};
        builder.setDirection(oboe::Direction::Output);
        builder.setPerformanceMode(oboe::PerformanceMode::LowLatency);
        builder.setUsage(oboe::Usage::Game);
        builder.setSharingMode(oboe::SharingMode::Shared);
        /* Don't let Oboe convert. We should be able to handle anything it gives
         * back.
         */
        builder.setSampleRateConversionQuality(oboe::SampleRateConversionQuality::None);
        builder.setChannelConversionAllowed(false);
        builder.setFormatConversionAllowed(false);
        builder.setCallback(this);

        if(mDevice.mFlags.test(DeviceFlag::FrequencyRequest))
        {
            builder.setSampleRateConversionQuality(oboe::SampleRateConversionQuality::High);
            builder.setSampleRate(gsl::narrow_cast<int32_t>(mDevice.mSampleRate));
        }
        if(mDevice.mFlags.test(DeviceFlag::ChannelsRequest))
        {
            /* Only use mono or stereo at user request. There's no telling what
             * other counts may be inferred as.
             */
            builder.setChannelCount((mDevice.FmtChans==DevFmtMono) ? oboe::ChannelCount::Mono
                                                                    : (mDevice.FmtChans==DevFmtStereo) ? oboe::ChannelCount::Stereo
                                                                                                        : oboe::ChannelCount::Unspecified);
        }
        if(mDevice.mFlags.test(DeviceFlag::SampleTypeRequest))
        {
            oboe::AudioFormat format{oboe::AudioFormat::Unspecified};
            switch(mDevice.FmtType)
            {
                case DevFmtByte:
                case DevFmtUByte:
                case DevFmtShort:
                case DevFmtUShort:
                    format = oboe::AudioFormat::I16;
                    break;
                case DevFmtInt:
                case DevFmtUInt:
#if OBOE_VERSION_MAJOR > 1 || (OBOE_VERSION_MAJOR == 1 && OBOE_VERSION_MINOR >= 6)
                    format = oboe::AudioFormat::I32;
                    break;
#endif
                case DevFmtFloat:
                    format = oboe::AudioFormat::Float;
                    break;
            }
            builder.setFormat(format);
        }

        auto result = builder.openStream(mStream);
        /* If the format failed, try asking for the defaults. */
        while(result == oboe::Result::ErrorInvalidFormat)
        {
            if(builder.getFormat() != oboe::AudioFormat::Unspecified)
                builder.setFormat(oboe::AudioFormat::Unspecified);
            else if(builder.getSampleRate() != oboe::kUnspecified)
                builder.setSampleRate(oboe::kUnspecified);
            else if(builder.getChannelCount() != oboe::ChannelCount::Unspecified)
                builder.setChannelCount(oboe::ChannelCount::Unspecified);
            else
                break;
            result = builder.openStream(mStream);
        }
        if(result != oboe::Result::OK)
        {
            mStream.reset();
            throw al::backend_exception{al::backend_error::DeviceError, "Failed to create stream: {}",
                                        oboe::convertToText(result)};
        }
        mStream->setBufferSizeInFrames(std::min(gsl::narrow_cast<int32_t>(mDevice.mBufferSize),
                                                mStream->getBufferCapacityInFrames()));
        TRACE("Got stream with properties:\n{}", oboe::convertToText(mStream.get()));

        if(std::cmp_not_equal(mStream->getChannelCount(), mDevice.channelsFromFmt()))
        {
            if(mStream->getChannelCount() >= 2)
                mDevice.FmtChans = DevFmtStereo;
            else if(mStream->getChannelCount() == 1)
                mDevice.FmtChans = DevFmtMono;
            else
                throw al::backend_exception{al::backend_error::DeviceError,
                                            "Got unhandled channel count: {}", mStream->getChannelCount()};
        }
        setDefaultWFXChannelOrder();

        switch(mStream->getFormat())
        {
            case oboe::AudioFormat::I16:
                mDevice.FmtType = DevFmtShort;
                break;
            case oboe::AudioFormat::Float:
                mDevice.FmtType = DevFmtFloat;
                break;
#if OBOE_VERSION_MAJOR > 1 || (OBOE_VERSION_MAJOR == 1 && OBOE_VERSION_MINOR >= 6)
            case oboe::AudioFormat::I32:
                mDevice.FmtType = DevFmtInt;
                break;
            case oboe::AudioFormat::I24:
#endif
#if OBOE_VERSION_MAJOR > 1 || (OBOE_VERSION_MAJOR == 1 && OBOE_VERSION_MINOR >= 8)
            case oboe::AudioFormat::IEC61937:
#endif
            case oboe::AudioFormat::Unspecified:
            case oboe::AudioFormat::Invalid:
                throw al::backend_exception{al::backend_error::DeviceError,
                                            "Got unhandled sample type: {}", oboe::convertToText(mStream->getFormat())};
        }
        mDevice.mSampleRate = gsl::narrow_cast<unsigned>(mStream->getSampleRate());

        /* Ensure the period size is no less than 10ms. It's possible for FramesPerCallback to be 0
         * indicating variable updates, but OpenAL should have a reasonable minimum update size set.
         * FramesPerBurst may not necessarily be correct, but hopefully it can act as a minimum
         * update size.
         */
        mDevice.mUpdateSize = std::max(mDevice.mSampleRate/100u,
                                        gsl::narrow_cast<unsigned>(mStream->getFramesPerBurst()));
        mDevice.mBufferSize = std::max(mDevice.mUpdateSize*2u,
                                        gsl::narrow_cast<unsigned>(mStream->getBufferSizeInFrames()));

        return true;
    }

    void OboePlayback::start()
    {
        std::lock_guard<std::mutex> lock{mStreamMutex};
        mPlaying = true;
        mReopenPending = false;

        /* The watch thread runs from the first start on */
        if(!mWatchThread.joinable())
            mWatchThread = std::thread{&OboePlayback::watchThreadProc, this};

        auto result = startStream();
        if(result != oboe::Result::OK)
        {
            /* Dead: disconnected while stopped (paused through a long sleep or a call), or closed
             * after an error.
             */
            WARN("Oboe stream can't be started ({}), opening it again", oboe::convertToText(result));
            result = reopenStream();
        }

        if(result != oboe::Result::OK)
        {
            /* Not thrown: OpenAL would disconnect the device for good. The watch thread keeps trying. */
            ERR("Failed to start the Oboe stream: {}, trying again in the background",
                oboe::convertToText(result));
            requestReopen();
        }
    }

    void OboePlayback::stop()
    {
        std::lock_guard<std::mutex> lock{mStreamMutex};
        mPlaying = false;
        mReopenPending = false;

        if(!mStream)
            return;

        /* A dead stream can't be stopped: start() opens it again */
        if(const auto result = mStream->stop(); result != oboe::Result::OK)
            ERR("Failed to stop stream: {}", oboe::convertToText(result));
    }


    struct OboeCapture final : BackendBase, oboe::AudioStreamCallback {
        using BackendBase::BackendBase;
        ~OboeCapture() override;
        std::shared_ptr<oboe::AudioStream> mStream;

        RingBufferPtr<std::byte> mRing;

        auto onAudioReady(oboe::AudioStream *oboeStream, void *audioData, int32_t numFrames)
        -> oboe::DataCallbackResult override;

        void open(std::string_view name) override;
        void start() override;
        void stop() override;
        void captureSamples(std::span<std::byte> outbuffer) override;
        auto availableSamples() -> std::size_t override;
    };

    OboeCapture::~OboeCapture()
    {
        if(mStream)
            mStream->close();
    }

    auto OboeCapture::onAudioReady(oboe::AudioStream*, void *const audioData, int32_t const numFrames)
    -> oboe::DataCallbackResult
    {
        std::ignore = mRing->write(std::span{static_cast<const std::byte*>(audioData),
                                             gsl::narrow_cast<uint32_t>(numFrames)*mRing->getElemSize()});
        return oboe::DataCallbackResult::Continue;
    }


    void OboeCapture::open(std::string_view name)
    {
        if(name.empty())
            name = GetDeviceName();
        else if(name != GetDeviceName())
            throw al::backend_exception{al::backend_error::NoDevice, "Device name \"{}\" not found",
                                        name};

        auto builder = oboe::AudioStreamBuilder{};
        builder.setDirection(oboe::Direction::Input)
                ->setPerformanceMode(oboe::PerformanceMode::LowLatency)
                ->setSampleRateConversionQuality(oboe::SampleRateConversionQuality::High)
                ->setChannelConversionAllowed(true)
                ->setFormatConversionAllowed(true)
                ->setSampleRate(gsl::narrow_cast<int32_t>(mDevice.mSampleRate))
                ->setCallback(this);
        /* Only use mono or stereo at user request. There's no telling what
         * other counts may be inferred as.
         */
        switch(mDevice.FmtChans)
        {
            case DevFmtMono:
                builder.setChannelCount(oboe::ChannelCount::Mono);
                break;
            case DevFmtStereo:
                builder.setChannelCount(oboe::ChannelCount::Stereo);
                break;
            case DevFmtQuad:
            case DevFmtX51:
            case DevFmtX61:
            case DevFmtX71:
            case DevFmtX714:
            case DevFmtX7144:
            case DevFmtX3D71:
            case DevFmtAmbi3D:
                throw al::backend_exception{al::backend_error::DeviceError, "{} capture not supported",
                                            DevFmtChannelsString(mDevice.FmtChans)};
        }

        /* FIXME: This really should support UByte, but Oboe doesn't. We'll need to
         * convert.
         */
        switch(mDevice.FmtType)
        {
            case DevFmtShort:
                builder.setFormat(oboe::AudioFormat::I16);
                break;
            case DevFmtFloat:
                builder.setFormat(oboe::AudioFormat::Float);
                break;
            case DevFmtInt:
#if OBOE_VERSION_MAJOR > 1 || (OBOE_VERSION_MAJOR == 1 && OBOE_VERSION_MINOR >= 6)
                builder.setFormat(oboe::AudioFormat::I32);
                break;
#endif
            case DevFmtByte:
            case DevFmtUByte:
            case DevFmtUShort:
            case DevFmtUInt:
                throw al::backend_exception{al::backend_error::DeviceError,
                                            "{} capture samples not supported", DevFmtTypeString(mDevice.FmtType)};
        }

        if(const auto result = builder.openStream(mStream); result != oboe::Result::OK)
            throw al::backend_exception{al::backend_error::DeviceError, "Failed to create stream: {}",
                                        oboe::convertToText(result)};

        TRACE("Got stream with properties:\n{}", oboe::convertToText(mStream.get()));

        /* Ensure a minimum ringbuffer size of 100ms. */
        mRing = RingBuffer<std::byte>::Create(
                std::max(mDevice.mBufferSize, mDevice.mSampleRate/10u),
                gsl::narrow_cast<unsigned>(mStream->getBytesPerFrame()), false);

        mDeviceName = name;
    }

    void OboeCapture::start()
    {
        if(const auto result = mStream->start(); result != oboe::Result::OK)
            throw al::backend_exception{al::backend_error::DeviceError, "Failed to start stream: {}",
                                        oboe::convertToText(result)};
    }

    void OboeCapture::stop()
    {
        if(const auto result = mStream->stop(); result != oboe::Result::OK)
            ERR("Failed to stop stream: {}", oboe::convertToText(result));
    }

    auto OboeCapture::availableSamples() -> std::size_t
    { return mRing->readSpace(); }

    void OboeCapture::captureSamples(std::span<std::byte> const outbuffer)
    { std::ignore = mRing->read(outbuffer); }

} // namespace

auto OboeBackendFactory::init() -> bool { return true; }

auto OboeBackendFactory::querySupport(BackendType const type) -> bool
{ return type == BackendType::Playback || type == BackendType::Capture; }

auto OboeBackendFactory::enumerate(BackendType const type) -> std::vector<std::string>
{
    switch(type)
    {
        case BackendType::Playback:
        case BackendType::Capture:
            return std::vector{std::string{GetDeviceName()}};
    }
    return {};
}

auto OboeBackendFactory::createBackend(DeviceBase &device, BackendType const type) -> BackendPtr
{
    if(type == BackendType::Playback)
        return BackendPtr{new OboePlayback{device}};
    if(type == BackendType::Capture)
        return BackendPtr{new OboeCapture{device}};
    return BackendPtr{};
}

auto OboeBackendFactory::getFactory() -> BackendFactory&
{
    static OboeBackendFactory factory{};
    return factory;
}