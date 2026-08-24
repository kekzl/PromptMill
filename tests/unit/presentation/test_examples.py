"""Tests for the per-category starter examples."""

from promptmill.domain.entities.role import RoleCategory
from promptmill.presentation.examples import EXAMPLES_BY_CATEGORY, examples_for

EXPECTED_PER_CATEGORY = 6


class TestExamples:
    """Tests for the example sets."""

    def test_every_category_has_examples(self) -> None:
        """No category falls back silently."""
        for category in RoleCategory:
            assert category in EXAMPLES_BY_CATEGORY

    def test_each_category_has_six_pairs(self) -> None:
        """The UI renders exactly six buttons."""
        for category in RoleCategory:
            assert len(examples_for(category)) == EXPECTED_PER_CATEGORY

    def test_labels_and_texts_are_non_empty(self) -> None:
        """Every pair has a button label and a prompt."""
        for category in RoleCategory:
            for label, text in examples_for(category):
                assert label.strip()
                assert text.strip()

    def test_audio_examples_are_not_video_examples(self) -> None:
        """The whole point: sets differ per category."""
        video = {text for _, text in examples_for(RoleCategory.VIDEO)}
        audio = {text for _, text in examples_for(RoleCategory.AUDIO)}
        assert not video & audio
