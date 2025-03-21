import pytest
import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys

from ckpt_streamer import apply_state_dict
from utils import temp_file


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.conv = nn.Conv2d(3, 6, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class ModelWithNesting(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = SimpleModel()
        self.output = nn.Linear(5, 1)

    def forward(self, x):
        x = self.base(x)
        x = self.output(x)
        return x


def test_apply_state_dict_basic():
    """Test basic functionality of apply_state_dict with matching state dict."""
    # Create source model and get its state dict
    source_model = SimpleModel()
    source_state_dict = source_model.state_dict()

    # Save to a temporary file and load with mmap=True
    with temp_file() as tmp:
        torch.save({"state_dict": source_state_dict}, tmp)
        loaded = torch.load(tmp, map_location="cpu", mmap=True)
        source_state_dict = loaded["state_dict"]

    # Create a target model with different parameter values
    target_model = SimpleModel()

    # Ensure that source and target models are different
    with torch.no_grad():
        for param in target_model.parameters():
            param.data = torch.randn_like(param.data)

    # Check that models have different parameters initially
    for (source_k, source_v), (target_k, target_v) in zip(
        source_model.state_dict().items(), target_model.state_dict().items()
    ):
        assert source_k == target_k
        # For small values, they might completely match, so set wider tolerance
        assert not torch.allclose(source_v, target_v, rtol=1e-3, atol=1e-3)

    # Apply state dict
    result = apply_state_dict(target_model, source_state_dict)

    # Check that models now have the same parameters
    for (source_k, source_v), (target_k, target_v) in zip(
        source_model.state_dict().items(), target_model.state_dict().items()
    ):
        assert source_k == target_k
        assert torch.allclose(source_v, target_v)

    # Check return value
    assert isinstance(result, _IncompatibleKeys)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0


def test_apply_state_dict_with_converter():
    """Test apply_state_dict with a converter function."""
    source_model = SimpleModel()
    source_state_dict = source_model.state_dict()

    # Save to a temporary file and load with mmap=True
    with temp_file() as tmp:
        torch.save({"state_dict": source_state_dict}, tmp)
        loaded = torch.load(tmp, map_location="cpu", mmap=True)
        source_state_dict = loaded["state_dict"]

    target_model = SimpleModel()

    # Explicitly change model parameters
    with torch.no_grad():
        for param in target_model.parameters():
            param.data = torch.randn_like(param.data)

    # Define a converter that scales all tensors by 2.0
    def scale_converter(root_module, current_module, tensor):
        return tensor * 2.0

    # Apply state dict with converter
    apply_state_dict(target_model, source_state_dict, converter=scale_converter)

    # Check that target parameters are scaled versions of source parameters
    for (source_k, source_v), (target_k, target_v) in zip(
        source_model.state_dict().items(), target_model.state_dict().items()
    ):
        assert source_k == target_k
        assert torch.allclose(source_v * 2.0, target_v)


def test_apply_state_dict_assign_mode():
    """Test apply_state_dict with assign=True, replacing parameters instead of copying."""
    source_model = SimpleModel()
    source_state_dict = source_model.state_dict()

    # Save to a temporary file and load with mmap=True
    with temp_file() as tmp:
        torch.save({"state_dict": source_state_dict}, tmp)
        loaded = torch.load(tmp, map_location="cpu", mmap=True)
        source_state_dict = loaded["state_dict"]

    target_model = SimpleModel()

    # Store original parameter objects
    original_params = {k: v for k, v in target_model.named_parameters()}

    # Apply state dict with assign=True
    apply_state_dict(target_model, source_state_dict, assign=True)

    # Check that parameters match but are different objects
    new_params = {k: v for k, v in target_model.named_parameters()}

    for k in original_params:
        # Values should match source model
        assert torch.allclose(new_params[k], source_state_dict[k])
        # But the parameter objects should be different from original
        assert original_params[k] is not new_params[k]


def test_apply_state_dict_missing_keys():
    """Test apply_state_dict with missing keys in the state dict."""
    source_model = SimpleModel()
    source_state_dict = source_model.state_dict()

    # Remove a key to create missing key scenario
    partial_state_dict = {k: v for k, v in source_state_dict.items() if "linear2" not in k}

    # Save to a temporary file and load with mmap=True
    with temp_file() as tmp:
        torch.save({"state_dict": partial_state_dict}, tmp)
        loaded = torch.load(tmp, map_location="cpu", mmap=True)
        partial_state_dict = loaded["state_dict"]

    target_model = SimpleModel()

    # Apply partial state dict with strict=False
    result = apply_state_dict(target_model, partial_state_dict, strict=False)

    # Check return value
    assert len(result.missing_keys) > 0
    assert all("linear2" in k for k in result.missing_keys)
    assert len(result.unexpected_keys) == 0

    # With strict=True it should raise an error
    with pytest.raises(RuntimeError) as excinfo:
        apply_state_dict(target_model, partial_state_dict, strict=True)
    assert "Missing key(s)" in str(excinfo.value)


def test_apply_state_dict_unexpected_keys():
    """Test apply_state_dict with unexpected keys in the state dict."""
    source_model = SimpleModel()
    source_state_dict = source_model.state_dict()

    # Add extra key to create unexpected key scenario
    extra_state_dict = source_state_dict.copy()
    extra_state_dict["extra.weight"] = torch.randn(5, 5)

    # Save to a temporary file and load with mmap=True
    with temp_file() as tmp:
        torch.save({"state_dict": extra_state_dict}, tmp)
        loaded = torch.load(tmp, map_location="cpu", mmap=True)
        extra_state_dict = loaded["state_dict"]

    target_model = SimpleModel()

    # Apply with strict=False
    result = apply_state_dict(target_model, extra_state_dict, strict=False)

    # Check return value
    assert len(result.unexpected_keys) == 1
    assert "extra.weight" in result.unexpected_keys

    # With strict=True it should raise an error
    with pytest.raises(RuntimeError) as excinfo:
        apply_state_dict(target_model, extra_state_dict, strict=True)
    assert "Unexpected key(s)" in str(excinfo.value)


def test_apply_state_dict_nested_model():
    """Test apply_state_dict with a nested model structure."""
    source_model = ModelWithNesting()
    source_state_dict = source_model.state_dict()

    # Save to a temporary file and load with mmap=True
    with temp_file() as tmp:
        torch.save({"state_dict": source_state_dict}, tmp)
        loaded = torch.load(tmp, map_location="cpu", mmap=True)
        source_state_dict = loaded["state_dict"]

    target_model = ModelWithNesting()

    # Explicitly change model parameters
    with torch.no_grad():
        for param in target_model.parameters():
            param.data = torch.randn_like(param.data)

    # Apply state dict
    result = apply_state_dict(target_model, source_state_dict)

    # Check that nested parameters match
    for (source_k, source_v), (target_k, target_v) in zip(
        source_model.state_dict().items(), target_model.state_dict().items()
    ):
        assert source_k == target_k
        assert torch.allclose(source_v, target_v)

    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0


def test_apply_state_dict_memory_limits():
    """Test apply_state_dict with different memory limits."""
    source_model = SimpleModel()
    source_state_dict = source_model.state_dict()

    # Save to a temporary file and load with mmap=True
    with temp_file() as tmp:
        torch.save({"state_dict": source_state_dict}, tmp)
        loaded = torch.load(tmp, map_location="cpu", mmap=True)
        source_state_dict = loaded["state_dict"]

    target_model = SimpleModel()

    # Explicitly change model parameters
    with torch.no_grad():
        for param in target_model.parameters():
            param.data = torch.randn_like(param.data)

    # Test with very small memory limit (should still work but might log warnings)
    result = apply_state_dict(target_model, source_state_dict, memory_limit_mb=1)

    # Check that it still applied correctly despite low memory limit
    for (source_k, source_v), (target_k, target_v) in zip(
        source_model.state_dict().items(), target_model.state_dict().items()
    ):
        assert source_k == target_k
        assert torch.allclose(source_v, target_v)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
