"""Tests for domain value objects."""

import pytest

from promptmill.domain.value_objects.prompt_request import (
    MAX_INPUT_LENGTH,
    MAX_TEMPERATURE,
    MAX_TOKENS,
    MIN_TEMPERATURE,
    MIN_TOKENS,
    PromptGenerationRequest,
)
from promptmill.domain.value_objects.prompt_result import PromptGenerationResult


class TestPromptGenerationRequest:
    """Tests for PromptGenerationRequest value object."""

    def test_valid_request(self) -> None:
        """Test creating a valid request."""
        request = PromptGenerationRequest(
            user_input="Generate a sunset scene",
            role_display_name="[Video] Wan2.1",
            temperature=0.7,
            max_tokens=256,
        )
        assert request.user_input == "Generate a sunset scene"
        assert request.role_display_name == "[Video] Wan2.1"
        assert request.temperature == 0.7
        assert request.max_tokens == 256

    def test_request_is_frozen(self) -> None:
        """Test that request is immutable."""
        request = PromptGenerationRequest(
            user_input="Test",
            role_display_name="[Video] Test",
            temperature=0.7,
            max_tokens=256,
        )
        with pytest.raises(AttributeError):
            request.user_input = "New"  # type: ignore

    def test_stripped_input(self) -> None:
        """Test stripped_input property."""
        request = PromptGenerationRequest(
            user_input="  spaces around  ",
            role_display_name="[Video] Test",
            temperature=0.7,
            max_tokens=256,
        )
        assert request.stripped_input == "spaces around"

    def test_empty_input_raises(self) -> None:
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PromptGenerationRequest(
                user_input="",
                role_display_name="[Video] Test",
                temperature=0.7,
                max_tokens=256,
            )

    def test_whitespace_only_input_raises(self) -> None:
        """Test that whitespace-only input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PromptGenerationRequest(
                user_input="   ",
                role_display_name="[Video] Test",
                temperature=0.7,
                max_tokens=256,
            )

    def test_input_too_long_raises(self) -> None:
        """Test that too-long input raises ValueError."""
        with pytest.raises(ValueError, match="maximum length"):
            PromptGenerationRequest(
                user_input="x" * (MAX_INPUT_LENGTH + 1),
                role_display_name="[Video] Test",
                temperature=0.7,
                max_tokens=256,
            )

    def test_empty_role_raises(self) -> None:
        """Test that empty role raises ValueError."""
        with pytest.raises(ValueError, match="Role display name cannot be empty"):
            PromptGenerationRequest(
                user_input="Test",
                role_display_name="",
                temperature=0.7,
                max_tokens=256,
            )

    def test_temperature_too_low_raises(self) -> None:
        """Test that temperature below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be"):
            PromptGenerationRequest(
                user_input="Test",
                role_display_name="[Video] Test",
                temperature=MIN_TEMPERATURE - 0.1,
                max_tokens=256,
            )

    def test_temperature_too_high_raises(self) -> None:
        """Test that temperature above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be"):
            PromptGenerationRequest(
                user_input="Test",
                role_display_name="[Video] Test",
                temperature=MAX_TEMPERATURE + 0.1,
                max_tokens=256,
            )

    def test_tokens_too_low_raises(self) -> None:
        """Test that tokens below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Max tokens must be"):
            PromptGenerationRequest(
                user_input="Test",
                role_display_name="[Video] Test",
                temperature=0.7,
                max_tokens=MIN_TOKENS - 1,
            )

    def test_tokens_too_high_raises(self) -> None:
        """Test that tokens above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Max tokens must be"):
            PromptGenerationRequest(
                user_input="Test",
                role_display_name="[Video] Test",
                temperature=0.7,
                max_tokens=MAX_TOKENS + 1,
            )

    def test_boundary_values(self) -> None:
        """Test boundary values are accepted."""
        # Minimum values
        request_min = PromptGenerationRequest(
            user_input="X",
            role_display_name="[Video] Test",
            temperature=MIN_TEMPERATURE,
            max_tokens=MIN_TOKENS,
        )
        assert request_min.temperature == MIN_TEMPERATURE
        assert request_min.max_tokens == MIN_TOKENS

        # Maximum values
        request_max = PromptGenerationRequest(
            user_input="X" * MAX_INPUT_LENGTH,
            role_display_name="[Video] Test",
            temperature=MAX_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        assert request_max.temperature == MAX_TEMPERATURE
        assert request_max.max_tokens == MAX_TOKENS


class TestPromptGenerationResult:
    """Tests for PromptGenerationResult value object."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        result = PromptGenerationResult(
            content="A beautiful sunset over the ocean",
            model_used="Dolphin 3.0 8B",
            role_used="Wan2.1",
        )
        assert "sunset" in result.content
        assert result.model_used == "Dolphin 3.0 8B"
        assert result.role_used == "Wan2.1"

    def test_result_is_frozen(self) -> None:
        """Test that result is immutable."""
        result = PromptGenerationResult(
            content="Test",
            model_used="Test Model",
            role_used="Test Role",
        )
        with pytest.raises(AttributeError):
            result.content = "New"  # type: ignore

    def test_char_count(self) -> None:
        """Test char_count property."""
        result = PromptGenerationResult(
            content="Hello World",  # 11 characters
            model_used="Test",
            role_used="Test",
        )
        assert result.char_count == 11

    def test_word_count(self) -> None:
        """Test word_count property."""
        result = PromptGenerationResult(
            content="This is a test prompt with seven words",
            model_used="Test",
            role_used="Test",
        )
        assert result.word_count == 8  # 8 words

    def test_str_representation(self) -> None:
        """Test string representation."""
        result = PromptGenerationResult(
            content="Test content",
            model_used="Test Model",
            role_used="Test Role",
        )
        s = str(result)
        assert "chars=" in s
        assert "words=" in s
        assert "model=" in s
