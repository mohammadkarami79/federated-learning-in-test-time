#!/usr/bin/env python3
"""
Quick fix for aggregation error in Kim et al., 2023 reproduction
================================================================

This script applies a quick fix to the server aggregation function
to handle the Float/Long casting error.
"""

import shutil
from pathlib import Path

def apply_aggregation_fix():
    """Apply the aggregation fix by replacing the problematic function"""
    
    # Read the current file
    main_file = Path("main_kim2023_reproduction.py")
    if not main_file.exists():
        print("❌ main_kim2023_reproduction.py not found")
        return False
    
    # Create backup
    backup_file = Path("main_kim2023_reproduction.py.backup")
    shutil.copy2(main_file, backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    # Read content
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Find and replace the problematic aggregate function
    old_aggregate = '''    def aggregate(self, client_states: List[Dict], client_weights: List[float]) -> Dict:
        """FedAvg aggregation"""
        global_state = self.global_model.state_dict()
        
        # Weighted average
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            
            for client_state, weight in zip(client_states, client_weights):
                global_state[key] += weight * client_state[key]
        
        self.global_model.load_state_dict(global_state)
        return global_state'''
    
    new_aggregate = '''    def aggregate(self, client_states: List[Dict], client_weights: List[float]) -> Dict:
        """FedAvg aggregation with proper type handling"""
        global_state = self.global_model.state_dict()
        
        # Weighted average with type safety
        for key in global_state.keys():
            # Skip non-trainable parameters
            if not global_state[key].requires_grad and global_state[key].dtype in [torch.long, torch.int]:
                continue
                
            # Initialize accumulator with proper type
            if global_state[key].dtype in [torch.long, torch.int]:
                # For integer tensors, use integer aggregation
                global_state[key] = torch.zeros_like(global_state[key], dtype=torch.long)
                for client_state, weight in zip(client_states, client_weights):
                    global_state[key] += (weight * client_state[key].float()).long()
            else:
                # For float tensors, use float aggregation
                global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
                for client_state, weight in zip(client_states, client_weights):
                    client_param = client_state[key]
                    if client_param.dtype != torch.float32:
                        client_param = client_param.float()
                    global_state[key] += weight * client_param
        
        self.global_model.load_state_dict(global_state)
        return global_state'''
    
    # Replace content
    if old_aggregate in content:
        content = content.replace(old_aggregate, new_aggregate)
        print("✅ Found and replaced old aggregate function")
    else:
        print("⚠️  Old aggregate function not found - applying new version")
        # If not found, we need to find a different pattern
        import re
        pattern = r'def aggregate\(self[^}]+return global_state'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content.replace(match.group(0), new_aggregate.strip())
            print("✅ Applied fix using pattern matching")
        else:
            print("❌ Could not locate aggregate function")
            return False
    
    # Write fixed content
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("✅ Aggregation fix applied successfully!")
    return True

if __name__ == "__main__":
    print("🔧 Applying aggregation fix for Kim et al., 2023...")
    
    success = apply_aggregation_fix()
    
    if success:
        print("\n✅ Fix applied! You can now restart the experiment:")
        print("nohup python run_kim2023_reproduction.py --background --log-suffix '_aggregation_fix' > kim2023_runner_agg_fix.log 2>&1 & echo $! > kim2023_pid.txt")
    else:
        print("\n❌ Fix failed. Please manually update the aggregate function.")
