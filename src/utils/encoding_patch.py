import builtins
import os

def apply_utf8_patch():
    """
    Applies an idempotent UTF-8 patch for Windows systems.
    Prevents RecursionError by checking if already patched.
    """
    if not hasattr(builtins, '_original_open_patched'):
        # Save the truly original open
        builtins._original_open_patched = builtins.open
        
        def smart_open(file, mode='r', *args, **kwargs):
            # Only apply encoding to text modes
            if 'b' not in mode:
                kwargs.setdefault('encoding', 'utf-8')
            return builtins._original_open_patched(file, mode, *args, **kwargs)
        
        builtins.open = smart_open
        return True
    return False
