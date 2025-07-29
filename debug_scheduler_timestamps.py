#!/usr/bin/env python3
"""
Create a debug version of scheduler files with logging to trace timestamp issues.
This script generates modified versions with debug prints.
"""

import os


def create_scheduler_debug_patch():
    """Create a debug patch for scheduler.py"""
    
    patch_content = '''
# Add this to scheduler.py around line 1244 (in handle_generate_request method)
# After: if add_to_grammar_queue:
#     req.queue_time_start = time.perf_counter()
#     self.grammar_queue.append(req)

if add_to_grammar_queue:
    req.queue_time_start = time.perf_counter()
    if self.enable_metrics:
        logger.info(f"[DEBUG] Set queue_time_start for req {req.rid}: {req.queue_time_start}")
    self.grammar_queue.append(req)
else:
    self._add_request_to_queue(req)

# Also modify _add_request_to_queue around line 1250:
def _add_request_to_queue(self, req: Req):
    req.queue_time_start = time.perf_counter()
    if self.enable_metrics:
        logger.info(f"[DEBUG] Set queue_time_start for req {req.rid}: {req.queue_time_start}")
    # ... rest of the method

# And around line 1762 (where queue_time_end is set):
if self.enable_metrics:
    # only record queue time when enable_metrics is True to avoid overhead
    for req in can_run_list:
        req.queue_time_end = time.perf_counter()
        logger.info(f"[DEBUG] Set queue_time_end for req {req.rid}: {req.queue_time_end}, duration: {req.queue_time_end - req.queue_time_start:.3f}s")
'''
    
    with open('scheduler_debug_patch.txt', 'w') as f:
        f.write(patch_content)
    
    print("✅ Created scheduler_debug_patch.txt")


def create_output_processor_debug_patch():
    """Create a debug patch for scheduler_output_processor_mixin.py"""
    
    patch_content = '''
# Add this to scheduler_output_processor_mixin.py in stream_output_generation method
# Around line 591 where queue times are collected:

# Collect queue time information
queue_start_val = req.queue_time_start if hasattr(req, 'queue_time_start') else None
queue_end_val = req.queue_time_end if hasattr(req, 'queue_time_end') else None

if hasattr(self, 'logger'):
    self.logger.info(f"[DEBUG] Collecting queue times for req {req.rid}: start={queue_start_val}, end={queue_end_val}")

queue_time_start.append(queue_start_val)
queue_time_end.append(queue_end_val)

# And before sending BatchTokenIDOut (around line 682):
if hasattr(self, 'logger'):
    self.logger.info(f"[DEBUG] Sending BatchTokenIDOut with {len(rids)} requests")
    self.logger.info(f"[DEBUG] queue_time_start list: {queue_time_start[:3]}...")  # First 3 values
    self.logger.info(f"[DEBUG] queue_time_end list: {queue_time_end[:3]}...")
    self.logger.info(f"[DEBUG] spec_verify_ct length: {len(spec_verify_ct)}")
'''
    
    with open('output_processor_debug_patch.txt', 'w') as f:
        f.write(patch_content)
    
    print("✅ Created output_processor_debug_patch.txt")


def create_tokenizer_manager_debug_patch():
    """Create a debug patch for tokenizer_manager.py"""
    
    patch_content = '''
# Add this to tokenizer_manager.py in _handle_batch_output method
# Around line 1417 where queue timestamps are added to meta_info:

# Add queue time tracking from scheduler
queue_start = recv_obj.queue_time_start[i] if hasattr(recv_obj, 'queue_time_start') and recv_obj.queue_time_start else None
queue_end = recv_obj.queue_time_end[i] if hasattr(recv_obj, 'queue_time_end') and recv_obj.queue_time_end else None

logger.info(f"[DEBUG] Processing request {rid}: queue_start={queue_start}, queue_end={queue_end}")

meta_info.update({
    # ... existing fields ...
    "queue_time_start": queue_start,
    "queue_time_end": queue_end,
})
'''
    
    with open('tokenizer_manager_debug_patch.txt', 'w') as f:
        f.write(patch_content)
    
    print("✅ Created tokenizer_manager_debug_patch.txt")


def create_apply_debug_script():
    """Create a script to help apply the debug patches."""
    
    script_content = '''#!/bin/bash
# Script to apply debug patches to SGLang

echo "This script will help you apply debug patches to SGLang"
echo "You need to manually edit the files based on the patch files created"
echo ""
echo "Files to modify:"
echo "1. python/sglang/srt/managers/scheduler.py"
echo "2. python/sglang/srt/managers/scheduler_output_processor_mixin.py"
echo "3. python/sglang/srt/managers/tokenizer_manager.py"
echo ""
echo "Instructions:"
echo "1. Open each file in your editor"
echo "2. Find the locations mentioned in the patch files"
echo "3. Add the debug logging code"
echo "4. Save and restart the SGLang servers"
echo ""
echo "After applying patches, run the servers with:"
echo "python -m sglang.launch_server ... --log-level info"
echo ""
echo "Then check the server logs for [DEBUG] messages"
'''
    
    with open('apply_debug_patches.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('apply_debug_patches.sh', 0o755)
    print("✅ Created apply_debug_patches.sh")


def main():
    print("="*60)
    print("Creating Debug Patches for SGLang")
    print("="*60)
    
    create_scheduler_debug_patch()
    create_output_processor_debug_patch()
    create_tokenizer_manager_debug_patch()
    create_apply_debug_script()
    
    print("\n" + "="*60)
    print("Debug patches created!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Review the patch files:")
    print("   - scheduler_debug_patch.txt")
    print("   - output_processor_debug_patch.txt")
    print("   - tokenizer_manager_debug_patch.txt")
    print("\n2. Apply the patches to your SGLang installation")
    print("3. Restart servers with --log-level info")
    print("4. Run a test and check server logs for [DEBUG] messages")
    print("\nThis will help identify where the timestamps are getting lost")


if __name__ == "__main__":
    main()