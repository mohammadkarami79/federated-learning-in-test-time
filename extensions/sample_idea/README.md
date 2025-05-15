# Sample Extension for Combined Defense

This is a template showing how to extend the combined DiffPure + pFedDef defense with your own ideas.

## How to Use This Template

1. **Inherit from CombinedClassifier**: Create a subclass that inherits from the base `CombinedClassifier` to leverage existing functionality while adding your improvements.

2. **Override the forward() method**: Implement your custom defense logic by overriding the forward method.

3. **Add a configuration flag**: Add a flag to enable/disable your extension in the configuration.

## Implementation Example

```python
# In defense_hook.py
from defense.combined_defense import CombinedClassifier

class NewDefenseHook(CombinedClassifier):
    def __init__(self, diffuser, pfeddef_model, cfg):
        super().__init__(diffuser, pfeddef_model, cfg)
        # Initialize any additional components needed for your defense
        self.enable_hook = getattr(cfg, 'ENABLE_NEW_HOOK', False)
        
    def forward(self, x, client_id=None):
        if not self.enable_hook:
            # Fall back to the original implementation if hook is disabled
            return super().forward(x, client_id)
            
        # Your custom implementation goes here
        # For example, add additional preprocessing, defense layers, etc.
        # ...
        
        # Then either:
        # 1. Call the parent's implementation with your modified input:
        # return super().forward(modified_x, client_id)
        # 
        # 2. Or implement a completely new defense approach:
        # logits = self.your_custom_defense(x)
        # return logits
```

## Integration Steps

1. Modify `config.py` to add your configuration flag:
   ```python
   # In config.py, add to Config class:
   ENABLE_NEW_HOOK: bool = False  # Enable the new defense hook
   
   # Add to both presets:
   'ENABLE_NEW_HOOK': False,  # or True to enable by default
   ```

2. Import and use your hook in your run script:
   ```python
   # In run_with_hook.py
   from extensions.sample_idea.defense_hook import NewDefenseHook
   
   # Use NewDefenseHook instead of CombinedClassifier
   classifier = NewDefenseHook(diffuser, pfeddef_model, cfg)
   ```

## Testing Your Extension

1. Run the quick test with your extension:
   ```
   python extensions/sample_idea/run_with_hook.py
   ```

2. Compare the results with and without your extension:
   ```
   python quick_test.py  # baseline
   ```

## Best Practices

1. **Modularity**: Keep your changes isolated in the extension.
2. **Fallback**: Provide a way to disable your extension for comparison.
3. **Documentation**: Document your approach and why it improves the defense.
4. **Metrics**: Add metrics to measure the specific improvements of your method. 