from __future__ import annotations

import asyncio
import contextlib
import datetime
import io
import wave

import numpy as np
import pyaudio
import pydub
import pytest
import pytest_asyncio

from audex import utils
from audex.lib.recorder import AudioConfig
from audex.lib.recorder import AudioFormat
from audex.lib.recorder import AudioRecorder


class TestAudioConfig:
    """Test AudioConfig configuration."""

    def test_default_config(self):
        """Test default audio configuration."""
        config = AudioConfig()
        assert config.format == pyaudio.paInt16
        assert config.channels == 1
        assert config.rate == 16000
        assert config.chunk == 1024
        assert config.input_device_index is None

    def test_custom_config(self):
        """Test custom audio configuration."""
        config = AudioConfig(
            format=pyaudio.paInt32,
            channels=2,
            rate=48000,
            chunk=2048,
            input_device_index=1,
        )
        assert config.format == pyaudio.paInt32
        assert config.channels == 2
        assert config.rate == 48000
        assert config.chunk == 2048
        assert config.input_device_index == 1


class TestAudioRecorder:
    """Test AudioRecorder functionality."""

    @pytest_asyncio.fixture
    async def recorder(self, mock_store):
        """Create a recorder instance."""
        recorder = AudioRecorder(
            store=mock_store,
            config=AudioConfig(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                chunk=1024,
            ),
        )
        await recorder.init()
        yield recorder
        await recorder.close()

    @pytest.mark.asyncio
    async def test_init_and_close(self, mock_store):
        """Test recorder initialization and cleanup."""
        recorder = AudioRecorder(store=mock_store)
        await recorder.init()

        assert recorder._audio is not None
        assert not recorder.is_recording
        assert recorder.current_segment_key is None

        await recorder.close()
        assert recorder._audio is None

    @pytest.mark.asyncio
    async def test_list_input_devices(self, recorder):
        """Test listing audio input devices."""
        devices = recorder.list_input_devices()
        assert isinstance(devices, list)
        # May be empty in CI environment
        for device in devices:
            assert "index" in device
            assert "name" in device
            assert "channels" in device
            assert "default_rate" in device

    @pytest.mark.asyncio
    async def test_format_mapping(self, mock_store):
        """Test that different audio formats map to correct dtypes."""
        test_cases = [
            (pyaudio.paInt8, np.int8, 1),
            (pyaudio.paInt16, np.int16, 2),
            (pyaudio.paInt32, np.int32, 4),
            (pyaudio.paFloat32, np.float32, 4),
        ]

        for fmt, expected_dtype, expected_width in test_cases:
            config = AudioConfig(format=fmt)
            recorder = AudioRecorder(store=mock_store, config=config)

            assert recorder._numpy_dtype == expected_dtype
            assert recorder._sample_width == expected_width

    @pytest.mark.asyncio
    async def test_unsupported_format(self, mock_store):
        """Test that unsupported audio format raises error."""
        with pytest.raises(ValueError, match="Unsupported audio format"):
            AudioRecorder(
                store=mock_store,
                config=AudioConfig(format=999),  # Invalid format
            )

    def test_find_frame_index_empty(self, recorder):
        """Test frame index search with no frames."""
        now = utils.utcnow()
        assert recorder._find_frame_index(now) == 0

    def test_find_frame_index_single(self, recorder):
        """Test frame index search with single frame."""
        now = utils.utcnow()
        recorder._frames_timestamps = [now]

        assert recorder._find_frame_index(now) == 0
        assert recorder._find_frame_index(now - datetime.timedelta(seconds=1)) == 0
        assert recorder._find_frame_index(now + datetime.timedelta(seconds=1)) == 0

    def test_find_frame_index_multiple(self, recorder):
        """Test frame index search with multiple frames."""
        base = utils.utcnow()
        timestamps = [base + datetime.timedelta(seconds=i) for i in range(10)]
        recorder._frames_timestamps = timestamps

        # Test exact matches
        assert recorder._find_frame_index(timestamps[0]) == 0
        assert recorder._find_frame_index(timestamps[5]) == 5
        assert recorder._find_frame_index(timestamps[9]) == 9

        # Test before first
        assert recorder._find_frame_index(base - datetime.timedelta(seconds=1)) == 0

        # Test after last
        assert recorder._find_frame_index(base + datetime.timedelta(seconds=20)) == 9

        # Test in-between (should return lower index)
        mid_time = timestamps[5] + datetime.timedelta(milliseconds=500)
        assert recorder._find_frame_index(mid_time) == 5

    def test_unpack_24bit(self, recorder):
        """Test 24-bit audio unpacking."""
        # Create test 24-bit data (3 bytes per sample)
        # Sample 1: 0x010203 = 66051
        # Sample 2: 0xFFFEFD = -259
        data = bytes([0x03, 0x02, 0x01, 0xFD, 0xFE, 0xFF])

        result = recorder._unpack_24bit(data)
        assert len(result) == 2
        assert result.dtype == np.int32

        # Verify values (with sign extension)
        expected = np.array([66051, -259], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_pack_24bit(self, recorder):
        """Test 24-bit audio packing."""
        # Create test 32-bit samples
        samples = np.array([66051, -259, 0], dtype=np.int32)

        result = recorder._pack_24bit(samples)
        assert len(result) == 9  # 3 samples * 3 bytes

        # Verify bytes
        assert result[0:3] == bytes([0x03, 0x02, 0x01])  # 66051
        assert result[3:6] == bytes([0xFD, 0xFE, 0xFF])  # -259
        assert result[6:9] == bytes([0x00, 0x00, 0x00])  # 0

    def test_resample_audio_numpy_same_rate(self, recorder):
        """Test resampling with same source and destination rate."""
        # Create test audio (1 second at 16kHz)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        audio = (audio * 32767).astype(np.int16)

        result = recorder._resample_audio_numpy(
            audio,
            src_rate=16000,
            dst_rate=16000,
            src_channels=1,
            dst_channels=1,
        )

        assert result.dtype == np.int16
        np.testing.assert_array_almost_equal(result, audio)

    def test_resample_audio_numpy_rate_change(self, recorder):
        """Test resampling with different rates."""
        # Create test audio (1 second at 16kHz)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        audio = (audio * 32767).astype(np.int16)

        # Downsample to 8kHz
        result = recorder._resample_audio_numpy(
            audio,
            src_rate=16000,
            dst_rate=8000,
            src_channels=1,
            dst_channels=1,
        )

        assert result.dtype == np.int16
        assert len(result) == 8000  # Half the length

    def test_resample_audio_numpy_mono_to_stereo(self, recorder):
        """Test converting mono to stereo."""
        audio = np.arange(100, dtype=np.int16)

        result = recorder._resample_audio_numpy(
            audio,
            src_rate=16000,
            dst_rate=16000,
            src_channels=1,
            dst_channels=2,
        )

        assert result.dtype == np.int16
        assert len(result) == 200  # Doubled for stereo

        # Every other sample should be the same (duplicated channels)
        assert np.array_equal(result[::2], result[1::2])

    def test_resample_audio_numpy_stereo_to_mono(self, recorder):
        """Test converting stereo to mono."""
        # Create interleaved stereo: [L1, R1, L2, R2, ...]
        audio = np.arange(200, dtype=np.int16)

        result = recorder._resample_audio_numpy(
            audio,
            src_rate=16000,
            dst_rate=16000,
            src_channels=2,
            dst_channels=1,
        )

        assert result.dtype == np.int16
        assert len(result) == 100  # Half the length

    def test_to_pydub_segment_int16(self, recorder):
        """Test converting numpy array to pydub AudioSegment."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        audio = (audio * 32767).astype(np.int16)

        segment = recorder._to_pydub_segment(audio, sample_rate=16000, channels=1)

        assert isinstance(segment, pydub.AudioSegment)
        assert segment.frame_rate == 16000
        assert segment.channels == 1
        assert segment.sample_width == 2
        assert len(segment) == 1000  # 1 second in milliseconds

    def test_to_pydub_segment_float32(self, mock_store):
        """Test converting float32 audio to pydub."""
        recorder = AudioRecorder(
            store=mock_store,
            config=AudioConfig(format=pyaudio.paFloat32),
        )

        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32))

        segment = recorder._to_pydub_segment(audio, sample_rate=16000, channels=1)

        assert isinstance(segment, pydub.AudioSegment)
        assert segment.frame_rate == 16000
        assert segment.channels == 1

    def test_encode_audio_pcm(self, recorder):
        """Test encoding audio to PCM format."""
        audio = np.arange(1000, dtype=np.int16)

        result = recorder._encode_audio(
            audio,
            sample_rate=16000,
            channels=1,
            output_format=AudioFormat.PCM,
        )

        assert isinstance(result, bytes)
        assert len(result) == 2000  # 1000 samples * 2 bytes
        assert result == audio.tobytes()

    def test_encode_audio_wav(self, recorder):
        """Test encoding audio to WAV format."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        audio = (audio * 32767).astype(np.int16)

        result = recorder._encode_audio(
            audio,
            sample_rate=16000,
            channels=1,
            output_format=AudioFormat.WAV,
        )

        assert isinstance(result, bytes)
        assert len(result) > len(audio.tobytes())  # WAV has header

        # Verify it's a valid WAV file
        wav_buffer = io.BytesIO(result)
        with wave.open(wav_buffer, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2

    def test_encode_audio_mp3(self, recorder):
        """Test encoding audio to MP3 format."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        audio = (audio * 32767).astype(np.int16)

        result = recorder._encode_audio(
            audio,
            sample_rate=16000,
            channels=1,
            output_format=AudioFormat.MP3,
        )

        assert isinstance(result, bytes)
        assert len(result) > 0
        # MP3 should be smaller than raw PCM
        assert len(result) < len(audio.tobytes())

        # Verify it's a valid MP3 by checking common signatures
        # MP3 files can start with ID3 tag (0x49 0x44 0x33) or frame sync (0xFF 0xFB/FA/F3/F2)
        is_id3 = result[0:3] == b"ID3"
        is_frame_sync = result[0] == 0xFF and (result[1] & 0xE0) == 0xE0

        assert is_id3 or is_frame_sync, f"Invalid MP3 header: {result[0:4].hex()}"

    @pytest.mark.asyncio
    async def test_clear_frames(self, recorder):
        """Test clearing recorded frames."""
        # Add some fake frames
        recorder._frames_data = [np.array([1, 2, 3], dtype=np.int16)]
        recorder._frames_timestamps = [utils.utcnow()]
        recorder._stream_position = 100

        recorder.clear_frames()

        assert len(recorder._frames_data) == 0
        assert len(recorder._frames_timestamps) == 0
        assert recorder._stream_position == 0

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_store):
        """Test using recorder as async context manager."""
        async with AudioRecorder(store=mock_store) as recorder:
            assert recorder._audio is not None

        assert recorder._audio is None


class TestAudioRecorderIntegration:
    """Integration tests with simulated recording."""

    @pytest.mark.asyncio
    async def test_simulated_recording(self, mock_store):
        """Test simulated recording without actual audio device."""
        recorder = AudioRecorder(
            store=mock_store,
            config=AudioConfig(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                chunk=1024,
            ),
        )
        await recorder.init()

        # Manually inject frames (simulate recording)
        recorder._is_recording = True
        recorder._started_at = utils.utcnow()
        recorder._current_key = "test/audio.wav"

        # Generate 1 second of audio (440Hz sine wave)
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        audio = (audio * 32767).astype(np.int16)

        # Split into chunks
        chunk_size = 1024
        base_time = recorder._started_at
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            timestamp = base_time + datetime.timedelta(seconds=i / sample_rate)
            recorder._frames_data.append(chunk)
            recorder._frames_timestamps.append(timestamp)

        # Test segment extraction
        start_time = base_time + datetime.timedelta(milliseconds=200)
        end_time = base_time + datetime.timedelta(milliseconds=800)

        segment_data = await recorder.segment(
            started_at=start_time,
            ended_at=end_time,
            format=AudioFormat.WAV,
        )

        assert isinstance(segment_data, bytes)
        assert len(segment_data) > 0

        # Verify WAV file
        wav_buffer = io.BytesIO(segment_data)
        with wave.open(wav_buffer, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000

        await recorder.close()

    @pytest.mark.asyncio
    async def test_simulated_streaming(self, mock_store):
        """Test simulated streaming without actual audio device."""
        recorder = AudioRecorder(
            store=mock_store,
            config=AudioConfig(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                chunk=1024,
            ),
        )
        await recorder.init()

        # Manually inject frames
        recorder._is_recording = True
        recorder._started_at = utils.utcnow()

        # Generate audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        audio = (audio * 32767).astype(np.int16)

        # Split into chunks
        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            recorder._frames_data.append(chunk)

        # Test streaming
        streamed_chunks = []
        stream_task = asyncio.create_task(
            self._collect_stream_chunks(
                recorder,
                chunk_size=4096,
                format=AudioFormat.PCM,
                max_chunks=3,
            )
        )

        # Wait a bit for streaming
        await asyncio.sleep(0.1)

        # Stop recording
        recorder._is_recording = False

        # Collect results
        with contextlib.suppress(TimeoutError):
            streamed_chunks = await asyncio.wait_for(stream_task, timeout=2.0)

        assert len(streamed_chunks) > 0
        for chunk in streamed_chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0

        await recorder.close()

    async def _collect_stream_chunks(
        self,
        recorder,
        chunk_size,
        format,
        max_chunks,
    ):
        """Helper to collect stream chunks."""
        chunks = []
        async for chunk in recorder.stream(chunk_size=chunk_size, format=format):
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break
        return chunks


class TestAudioFormat:
    """Test AudioFormat enum."""

    def test_format_values(self):
        """Test audio format enum values."""
        assert AudioFormat.PCM.value == "pcm"
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.OPUS.value == "opus"

    def test_format_from_string(self):
        """Test creating format from string."""
        assert AudioFormat("pcm") == AudioFormat.PCM
        assert AudioFormat("wav") == AudioFormat.WAV
        assert AudioFormat("mp3") == AudioFormat.MP3
        assert AudioFormat("opus") == AudioFormat.OPUS
