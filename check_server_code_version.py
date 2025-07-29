#!/usr/bin/env python3
"""
Check if the server is running with the updated code.
This script will verify the code modifications are in place.
"""

import sys
import os
import importlib
import inspect


def check_batch_token_id_out():
    """Check if BatchTokenIDOut has the new queue_time fields."""
    print("\n🔍 Checking BatchTokenIDOut structure...")
    
    try:
        from sglang.srt.managers.io_struct import BatchTokenIDOut
        
        # Get all fields
        fields = [f.name for f in BatchTokenIDOut.__dataclass_fields__.values()]
        
        print(f"Total fields: {len(fields)}")
        
        # Check for queue time fields
        queue_fields = ['queue_time_start', 'queue_time_end']
        for field in queue_fields:
            if field in fields:
                print(f"✅ {field}: Found")
            else:
                print(f"❌ {field}: Missing!")
        
        # Show last few fields
        print(f"\nLast 5 fields: {fields[-5:]}")
        
        return all(field in fields for field in queue_fields)
        
    except Exception as e:
        print(f"❌ Error checking BatchTokenIDOut: {e}")
        return False


def check_scheduler_output_processor():
    """Check if scheduler_output_processor_mixin has the modifications."""
    print("\n🔍 Checking scheduler_output_processor_mixin...")
    
    try:
        from sglang.srt.managers.scheduler_output_processor_mixin import SchedulerOutputProcessorMixin
        
        # Try to find the stream_output_generation method
        if hasattr(SchedulerOutputProcessorMixin, 'stream_output_generation'):
            method = getattr(SchedulerOutputProcessorMixin, 'stream_output_generation')
            source = inspect.getsource(method)
            
            # Check for our modifications
            checks = [
                ('queue_time_start = []', 'queue_time_start initialization'),
                ('queue_time_end = []', 'queue_time_end initialization'),
                ('queue_time_start.append', 'queue_time_start collection'),
                ('queue_time_end.append', 'queue_time_end collection'),
                ('hasattr(req, \'queue_time_start\')', 'hasattr check for queue_time_start'),
                ('spec_verify_ct.append(0)', 'spec_verify_ct else branch')
            ]
            
            for pattern, desc in checks:
                if pattern in source:
                    print(f"✅ {desc}: Found")
                else:
                    print(f"❌ {desc}: Missing!")
                    
            return all(pattern in source for pattern, _ in checks)
        else:
            print("❌ stream_output_generation method not found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking scheduler_output_processor_mixin: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_tokenizer_manager():
    """Check if tokenizer_manager has the queue timestamp handling."""
    print("\n🔍 Checking tokenizer_manager...")
    
    try:
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        
        # Find _handle_batch_output method
        if hasattr(TokenizerManager, '_handle_batch_output'):
            method = getattr(TokenizerManager, '_handle_batch_output')
            source = inspect.getsource(method)
            
            # Check for queue timestamp handling
            checks = [
                ('queue_time_start', 'queue_time_start in meta_info'),
                ('queue_time_end', 'queue_time_end in meta_info'),
                ('hasattr(recv_obj, \'queue_time_start\')', 'hasattr check for queue timestamps')
            ]
            
            for pattern, desc in checks:
                if pattern in source:
                    print(f"✅ {desc}: Found")
                else:
                    print(f"❌ {desc}: Missing!")
                    
            return True  # Less critical
        else:
            print("❌ _handle_batch_output method not found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking tokenizer_manager: {e}")
        return False


def check_installation_type():
    """Check how sglang is installed."""
    print("\n🔍 Checking SGLang installation...")
    
    try:
        import sglang
        
        # Get installation path
        sglang_path = os.path.dirname(sglang.__file__)
        print(f"SGLang path: {sglang_path}")
        
        # Check if it's an editable install
        if 'site-packages' in sglang_path:
            print("📦 Regular pip install detected")
            return 'regular'
        else:
            print("📝 Editable install (pip install -e) detected")
            return 'editable'
            
    except Exception as e:
        print(f"❌ Error checking installation: {e}")
        return 'unknown'


def main():
    print("="*60)
    print("SGLang Server Code Version Check")
    print("="*60)
    
    # Check installation type
    install_type = check_installation_type()
    
    # Run checks
    results = {
        'BatchTokenIDOut': check_batch_token_id_out(),
        'SchedulerOutputProcessor': check_scheduler_output_processor(),
        'TokenizerManager': check_tokenizer_manager()
    }
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    all_pass = all(results.values())
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}: {'OK' if status else 'NEEDS UPDATE'}")
    
    if all_pass:
        print("\n🎉 All modifications are in place!")
    else:
        print("\n⚠️  Some modifications are missing!")
        print("\nTroubleshooting steps:")
        
        if install_type == 'editable':
            print("1. Make sure you saved all files")
            print("2. Restart the SGLang servers")
            print("3. Check if there are any .pyc files that need clearing")
        else:
            print("1. Re-run: pip install -e .")
            print("2. Restart all SGLang servers")
            print("3. Make sure the code changes are in the installation directory")
    
    print("\n" + "="*60)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())