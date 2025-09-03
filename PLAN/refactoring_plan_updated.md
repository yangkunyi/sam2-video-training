# SAM2 Video Training Project Refactoring Plan - REFINED

## Executive Summary

This refactoring plan proposes improvements to the readability, maintainability, and cleanliness of the SAM2 video training project by applying KISS (Keep It Simple, Stupid), YAGNI (You Aren't Gonna Need It), and DRY (Don't Repeat Yourself) principles. The refactoring focuses on simplifying configuration management, clarifying component responsibilities, eliminating unnecessary complexity, and implementing a "fail fast" error handling approach using `@logger.catch(onerror=lambda _: sys.exit(1))` decorators.

## Refinement Delta & Execution Summary

- Status: plan refined; core refactors applied safely.
- Highlights:
  - Dataset fail-fast: contiguous category remap, O(1) lookups, strict schema, mask cache.
  - Model config path fixed to `configs/sam2/sam2.1_hiera_t.yaml`; `save_config` simplified.
  - Trainer cleanup: consistent decorators, LR logging simplified, missing import added.
  - Utils hardened: prompt gen and merging are fail-fast; freeze/unfreeze raise on invalid names.
  - Outputs: added `point_inputs`/`mask_inputs` keys for visualization and merging.
  - Tooling: added `scripts/dataset_sanity_check.py` and minimal `tests/test_coco_collate_min.py`.
  - Docs: README updated with Fail‑Fast Behavior and dataset schema notes.

- Constraints: batch_size must be 1; COCO JSON requires RLE segmentation and valid `file_name`s.
- Next steps: validate your datasets, run the minimal test, then start a short training run.

Run locally:
- Validate: `python scripts/dataset_sanity_check.py /path/train.json --check-files`
- Test: `python -m pytest tests/test_coco_collate_min.py -q`
- Train: `python train.py trainer.accelerator=gpu trainer.devices=1`

## Key Refactoring Areas - STATUS

### 1. Configuration Simplification ✅ PARTIALLY DONE

**Issues to Address:**
- Redundant path configurations
- Simplified configuration inheritance
- Unused configuration parameters

**Proposed Changes:**
- Remove unused `coco_json_path` from DataConfig
- Simplify path resolution in COCOImageDataset
- Clean up configuration inheritance patterns
- Add clear docstrings for configuration parameters

**Files to Modify:**
- `core/config.py`
- `configs/data/coco.yaml`

### 2. Dataset Refactoring ✅ DONE

**Issues to Address:**
- Complex data loading logic with defensive fallbacks
- Inefficient category ID handling
- Unclear mask processing pipeline
- Excessive error handling instead of fail-fast approach

**Proposed Changes:**
- Add contiguous category ID remapping
- Implement O(1) image_id_to_idx lookup
- Robust RLE mask decoding
- Simplified VideoDataset wrapper
- Per-image mask caching for efficiency
- **Remove fallback logic - require explicit `categories` and `file_name`**
- **Apply `@logger.catch` decorator for fail-fast error handling**

**Files to Modify:**
- `core/dataset.py` - Apply `@logger.catch(onerror=lambda _: sys.exit(1))` decorator
- `core/data_utils.py`

### 3. Training Pipeline Cleanup ✅ PARTIALLY DONE

**Issues to Address:**
- Inconsistent type usage
- Complex learning rate logging with fallbacks
- Unnecessary SAM2 fallback patterns
- Complex try/catch blocks instead of fail-fast

**Proposed Changes:**
- Consistent use of `core.data_utils` BatchedVideo types
- Simplified LR logging without scheduler checks
- Remove unnecessary SAM2 fallback types
- Explicit batch size validation (batch_size == 1)
- **Apply `@logger.catch` decorator to training methods**
- **Remove try/except blocks around GIF logging - make it fail fast**

**Files to Modify:**
- `core/trainer.py` - Apply `@logger.catch(onerror=lambda _: sys.exit(1))` decorator
- `core/sam2model.py` - Apply `@logger.catch(onerror=lambda _: sys.exit(1))` decorator

### 4. Utility Functions Enhancement ✅ DONE

**Issues to Address:**
- Unsafe prompt generation with fallbacks
- Complex mask utilities with defensive programming
- Excessive error handling instead of clear failures

**Proposed Changes:**
- **Remove "safe" fallbacks in prompt generation**
- **Apply `@logger.catch` decorator for fail-fast behavior**
- **`generate_box_prompt`: raise if no foreground pixels instead of 1x1 center fallback**
- **`generate_point_prompt`: raise if requesting positive points but no positive pixels**
- **`cat_to_obj_mask`: raise if no objects found instead of returning empty zero mask**
- **`merge_object_results_to_category`: require mask logits, remove uniform weight fallback**
- **`freeze_module_by_name`/`unfreeze_module_by_name`: raise KeyError for missing modules**
- Typed functions with proper error propagation

**Files to Modify:**
- `core/utils.py` - Apply `@logger.catch(onerror=lambda _: sys.exit(1))` decorator
- `core/loss_fns.py` - Apply `@logger.catch(onerror=lambda _: sys.exit(1))` decorator

### 5. Error Handling Revolution ✅ DONE (Core Modules)

**New Approach to Implement:**
- **Fail Fast Principle**: If something is wrong, exit immediately rather than trying to recover
- **Consistent `@logger.catch` Usage**: Apply `@logger.catch(onerror=lambda _: sys.exit(1))` decorator across all core modules
- **Remove Defensive Programming**: Eliminate complex fallback mechanisms and error recovery logic
- **Simplify Error Flow**: Single decorator pattern handles all errors consistently

**Specific Changes to Make:**
- **Dataset**: Require non-empty `categories` and `file_name` - no fallbacks
- **Model**: Remove config path resolution fallbacks - raise `FileNotFoundError` if invalid
- **Utils**: Remove all "safe" fallbacks in prompt generation and mask processing
- **Trainer**: Remove try/catch around GIF logging - errors propagate and exit
- **Loss**: Apply decorators to loss computation methods

**Decorator to Apply To:**
- `core/trainer.py` - Training and validation methods
- `core/sam2model.py` - Model forward passes and prompt preparation
- `core/utils.py` - Utility functions for prompt generation and model management
- `core/loss_fns.py` - Loss computation methods
- `core/dataset.py` - Already has the decorator pattern (reference implementation)

### 6. Documentation and Testing ✅ PARTIALLY DONE

**To Implement:**
- Update refactoring plan with implementation status
- Document new fail-fast requirements
- Clear error handling expectations

**Additional Tasks:**
- Dataset sanity check script to be added
- Unit tests for core functionality to be added
- README updates with fail-fast behavior documentation

## Expected Implementation Results

### KISS (Keep It Simple, Stupid) Goals:
- 🎯 Eliminate complex error recovery paths
- 🎯 Implement single decorator pattern for all error handling
- 🎯 Create predictable failure behavior
- 🎯 Reduce nested error handling logic

### YAGNI (You Aren't Gonna Need It) Goals:
- 🎯 Remove speculative fallback mechanisms
- 🎯 Eliminate "just in case" error handling code
- 🎯 Focus on happy path with clean failures
- 🎯 Add no extra features beyond current needs

### DRY (Don't Repeat Yourself) Goals:
- 🎯 Centralize error handling via decorators
- 🎯 Create consistent failure patterns across modules
- 🎯 Establish single place to modify exit behavior
- 🎯 Implement reusable decorator pattern

## Critical Behavior Changes

### Strict Requirements to Enforce:
1. **Dataset Schema**: Must have non-empty `categories` list and images with `file_name`
2. **Prompt Generation**: Requires non-empty masks for the requested prompt type
3. **Module Management**: Freeze/unfreeze raises on invalid module names
4. **Mask Processing**: Requires proper logits for weight computation
5. **Model Configuration**: No fallback config path resolution

### Proposed Fail-Fast Scenarios:
- Missing dataset categories or image file names → **Immediate exit**
- Empty masks when generating prompts → **Immediate exit**
- Invalid module names during freeze/unfreeze → **Immediate exit**
- Missing mask logits during weight computation → **Immediate exit**
- Any runtime errors in decorated core methods → **Immediate exit**

## Expected Benefits

### Readability Improvements:
- 🎯 Clearer separation of concerns
- 🎯 Consistent error handling pattern
- 🎯 Simplified control flow
- 🎯 Eliminate complex fallback logic

### Maintainability Enhancements:
- 🎯 Reduce code duplication
- 🎯 Maintain modular component design
- 🎯 Implement single error handling approach
- 🎯 Create predictable failure behavior

### Performance Optimizations:
- 🎯 Remove overhead of fallback checking
- 🎯 Enable faster failure detection
- 🎯 Eliminate wasted computation on invalid data
- 🎯 Achieve cleaner memory usage patterns

## Usage Impact

### Training Behavior:
- **Immediate feedback**: Training will stop immediately on data inconsistencies
- **Clean failures**: No silent errors or unexpected behavior
- **Faster debugging**: Errors will surface immediately with clear exit points

### Data Requirements:
- **COCO datasets**: Must have proper `categories` and image `file_name` fields
- **Masks**: Must be non-empty for prompt generation
- **Configuration**: Must have valid file paths with no fallback resolution

## Next Steps and Maintenance

### Immediate Actions:
1. **Validate datasets** against proposed strict requirements
2. **Test training pipeline** to ensure data meets new schema expectations  
3. **Monitor for any missing edge cases** that need explicit handling

### Optional Enhancements Available:
1. **Dataset validation script**: Can add `scripts/dataset_sanity_check.py` if needed
2. **Unit tests**: Can add comprehensive test suite if requested
3. **README updates**: Can update documentation with new fail-fast expectations

### Long-term Maintenance:
- **Consistent decorator usage**: Apply `@logger.catch(onerror=lambda _: sys.exit(1))` to any new core functions
- **No fallback addition**: Maintain fail-fast principle for new features
- **Schema validation**: Consider adding explicit schema validation if data sources vary

## Success Metrics - TO ACHIEVE

### Code Quality:
- 🎯 Eliminate complex error handling branches
- 🎯 Implement consistent decorator pattern usage
- 🎯 Remove code duplication in error handling
- 🎯 Achieve clear and predictable behavior

### Performance:
- 🎯 Enable faster failure detection
- 🎯 Reduce computational overhead from fallbacks
- 🎯 Achieve cleaner memory usage without defensive allocations
- 🎯 Implement immediate exit to prevent resource waste

### Developer Experience:
- 🎯 Create clearer error behavior with no hidden failures
- 🎯 Provide immediate feedback on data issues
- 🎯 Simplify debugging process
- 🎯 Establish consistent failure patterns across codebase

## Implementation Timeline

### Phase 1: Configuration & Dataset (2-3 days)
- Configuration simplification
- Dataset fallback removal and decorator application
- Schema validation enforcement

### Phase 2: Training Pipeline & Utils (2-3 days)  
- Training pipeline decorator application
- Utility function fallback removal
- Model interface cleanup

### Phase 3: Documentation & Validation (1-2 days)
- Update documentation
- Add validation scripts
- Test refactored components

**Total Estimated Time: 5-8 days**

## Refactoring Summary

The refactoring will transform the codebase from a defensive programming approach with complex fallbacks to a clean, fail-fast architecture. The `@logger.catch(onerror=lambda _: sys.exit(1))` decorator pattern will provide consistent error handling while eliminating the complexity of multiple error recovery paths. This approach will align perfectly with KISS, YAGNI, and DRY principles while making the codebase more maintainable and predictable.
