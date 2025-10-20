"""
Synthetic Evaluation Tasks for Architecture Validation.

These tasks test specific capabilities on untrained/randomly-initialized models
to provide directional signal about architectural differences.
"""

import torch
import torch.nn as nn
from typing import Any


class SyntheticMemoryTask:
    """
    Memory retrieval task: Can the model remember specific tokens from earlier segments?

    Task design:
    - Insert unique "marker" tokens at known positions in early segments
    - Query for those markers in later segments
    - Measure retrieval accuracy

    This tests:
    - Memory capacity (can it store information across segments?)
    - Memory augmentation (does coprocessor improve retrieval?)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        num_markers: int = 10,
        marker_start_id: int = 1000,
    ):
        """
        Initialize memory task.

        Args:
            vocab_size: Size of vocabulary
            num_markers: Number of unique markers to place
            marker_start_id: Starting vocab ID for markers (to ensure uniqueness)
        """
        self.vocab_size = vocab_size
        self.num_markers = num_markers
        self.marker_start_id = marker_start_id

        # Create unique marker tokens
        self.markers = list(range(marker_start_id, marker_start_id + num_markers))

    def generate_data(
        self,
        batch_size: int,
        segment_length: int,
        num_segments: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
        """
        Generate memory task data.

        Returns:
            context_segments: List of context segments with markers embedded
            query_segment: Query segment that asks about markers
            metadata: Information about marker placements
        """
        # Generate random base segments
        segments = []
        marker_positions = {}

        for seg_idx in range(num_segments):
            segment = torch.randint(
                0, self.vocab_size, (batch_size, segment_length), device=device
            )

            # Embed markers in early segments
            if seg_idx < self.num_markers:
                marker_id = self.markers[seg_idx]
                # Place marker at a specific position (e.g., middle of segment)
                marker_pos = segment_length // 2
                segment[:, marker_pos] = marker_id

                marker_positions[seg_idx] = {
                    "marker_id": marker_id,
                    "position": marker_pos,
                    "segment": seg_idx,
                }

            segments.append(segment)

        # Create query segment that asks about markers
        # For simplicity: query is asking "what was marker at segment X?"
        # We'll measure if logits predict the correct marker
        query_segment = torch.randint(
            0, self.vocab_size, (batch_size, segment_length), device=device
        )

        metadata = {"marker_positions": marker_positions, "markers": self.markers}

        return segments, query_segment, metadata

    def evaluate_memory_retrieval(
        self,
        logits: torch.Tensor,
        metadata: dict[str, Any],
        device: torch.device,
    ) -> float:
        """
        Evaluate memory retrieval accuracy.

        For each marker, check if the model assigns high probability to the correct marker token.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            metadata: Marker placement information
            device: Device

        Returns:
            Retrieval accuracy (0-100)
        """
        markers = metadata["markers"]
        batch_size = logits.shape[0]

        # For simplicity: check if any position in output has high probability for our markers
        # In reality, you'd want specific query positions, but this gives directional signal

        # Get probabilities for marker tokens
        probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

        # Extract probabilities for marker tokens
        marker_probs = probs[:, :, markers]  # [batch, seq_len, num_markers]

        # Measure: does the model assign higher-than-random probability to markers?
        # Random baseline would be num_markers / vocab_size
        random_baseline = len(markers) / self.vocab_size

        # Max probability assigned to any marker position
        max_marker_prob = marker_probs.max(dim=-1)[0]  # [batch, seq_len]
        avg_max_prob = max_marker_prob.mean().item()

        # Normalize to 0-100 scale
        # If model is random: ~random_baseline probability
        # If model is perfect: 1.0 probability
        # Scale to percentage
        retrieval_score = (avg_max_prob - random_baseline) / (1.0 - random_baseline)
        retrieval_score = max(0.0, min(1.0, retrieval_score))  # Clamp to [0, 1]

        return retrieval_score * 100


class SyntheticReasoningTask:
    """
    Pattern completion task: Can the model identify and continue patterns?

    Task design:
    - Create simple patterns (e.g., "A B A B" → predict "A B")
    - Test if model can recognize and continue the pattern
    - Measure pattern completion accuracy

    This tests:
    - Sequential reasoning
    - Pattern recognition
    - Deliberative thinking (coprocessor should help)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        pattern_start_id: int = 2000,
        num_patterns: int = 5,
    ):
        """
        Initialize reasoning task.

        Args:
            vocab_size: Size of vocabulary
            pattern_start_id: Starting vocab ID for pattern tokens
            num_patterns: Number of different patterns to test
        """
        self.vocab_size = vocab_size
        self.pattern_start_id = pattern_start_id
        self.num_patterns = num_patterns

        # Define pattern tokens
        self.pattern_tokens = list(
            range(pattern_start_id, pattern_start_id + num_patterns)
        )

    def generate_data(
        self,
        batch_size: int,
        segment_length: int,
        num_segments: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Generate reasoning task data.

        Returns:
            context_segments: Segments containing pattern demonstrations
            query_segment: Segment with incomplete pattern
            target_labels: Expected pattern completion
        """
        # Create segments with repeating patterns
        segments = []
        pattern = self.pattern_tokens[:3]  # Use first 3 tokens as pattern: A B C

        for seg_idx in range(num_segments):
            segment = torch.zeros((batch_size, segment_length), dtype=torch.long, device=device)

            # Fill with repeating pattern
            for i in range(segment_length):
                segment[:, i] = pattern[i % len(pattern)]

            # Add some noise (replace 20% with random tokens)
            noise_mask = torch.rand(batch_size, segment_length, device=device) < 0.2
            noise_tokens = torch.randint(
                0, self.vocab_size, (batch_size, segment_length), device=device
            )
            segment[noise_mask] = noise_tokens[noise_mask]

            segments.append(segment)

        # Query segment: start pattern, model should continue it
        query_segment = torch.zeros(
            (batch_size, segment_length), dtype=torch.long, device=device
        )

        # First half: show pattern
        pattern_length = len(pattern)
        for i in range(segment_length // 2):
            query_segment[:, i] = pattern[i % pattern_length]

        # Second half: model should predict pattern continuation
        target_labels = torch.zeros(
            (batch_size, segment_length), dtype=torch.long, device=device
        )
        for i in range(segment_length):
            target_labels[:, i] = pattern[i % pattern_length]

        return segments, query_segment, target_labels

    def evaluate_pattern_completion(
        self,
        logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> float:
        """
        Evaluate pattern completion accuracy.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            target_labels: Expected pattern tokens [batch, seq_len]

        Returns:
            Pattern completion accuracy (0-100)
        """
        # Get predictions
        predictions = logits.argmax(dim=-1)  # [batch, seq_len]

        # Focus on second half (where pattern should be predicted)
        seq_len = predictions.shape[1]
        second_half_start = seq_len // 2

        pred_second_half = predictions[:, second_half_start:]
        target_second_half = target_labels[:, second_half_start:]

        # Calculate accuracy
        correct = (pred_second_half == target_second_half).float()
        accuracy = correct.mean().item() * 100

        return accuracy


class MultiHopReasoningTask:
    """
    Multi-hop reasoning: Combine information from multiple segments.

    Task design:
    - Segment 1: "Key A is 123"
    - Segment 3: "Key B is 456"
    - Query: "What is A + B?" → Should output 579

    This tests:
    - Long-range memory
    - Arithmetic reasoning
    - Information integration
    """

    def __init__(self, vocab_size: int = 32000, num_hops: int = 3):
        """
        Initialize multi-hop reasoning task.

        Args:
            vocab_size: Size of vocabulary
            num_hops: Number of information pieces to combine
        """
        self.vocab_size = vocab_size
        self.num_hops = num_hops

        # Use specific token IDs for numbers (simplified)
        # In reality, you'd want proper tokenization
        self.number_token_start = 100

    def generate_data(
        self,
        batch_size: int,
        segment_length: int,
        num_segments: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
        """
        Generate multi-hop reasoning data.

        Returns:
            context_segments: Segments with factual information
            query_segment: Query asking to combine information
            metadata: Information about what should be retrieved
        """
        # For simplicity: embed numbers at specific positions
        segments = []
        facts = {}

        for seg_idx in range(num_segments):
            segment = torch.randint(
                0, self.vocab_size, (batch_size, segment_length), device=device
            )

            # Embed a "fact" in some segments
            if seg_idx < self.num_hops:
                fact_value = seg_idx + 1  # Simple values: 1, 2, 3, ...
                fact_token = self.number_token_start + fact_value
                fact_position = segment_length // 2

                segment[:, fact_position] = fact_token
                facts[f"fact_{seg_idx}"] = {
                    "value": fact_value,
                    "token": fact_token,
                    "segment": seg_idx,
                    "position": fact_position,
                }

            segments.append(segment)

        # Query segment asks to combine facts
        query_segment = torch.randint(
            0, self.vocab_size, (batch_size, segment_length), device=device
        )

        metadata = {"facts": facts, "expected_sum": sum(f["value"] for f in facts.values())}

        return segments, query_segment, metadata

    def evaluate_multi_hop_reasoning(
        self,
        logits: torch.Tensor,
        metadata: dict[str, Any],
    ) -> float:
        """
        Evaluate multi-hop reasoning.

        Args:
            logits: Model output logits
            metadata: Information about expected answer

        Returns:
            Reasoning accuracy (0-100)
        """
        # Check if model predicts expected sum token
        expected_sum = metadata["expected_sum"]
        expected_token = self.number_token_start + expected_sum

        probs = torch.softmax(logits, dim=-1)
        expected_prob = probs[:, :, expected_token].max().item()

        # Normalize to 0-100 scale
        # Random baseline: 1/vocab_size
        random_baseline = 1.0 / self.vocab_size
        score = (expected_prob - random_baseline) / (1.0 - random_baseline)
        score = max(0.0, min(1.0, score))

        return score * 100
