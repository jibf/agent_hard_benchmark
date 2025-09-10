import json
import glob
import os
from collections import defaultdict

def load_json_file(filepath):
    """Load and parse a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_task_details():
    """Extract detailed task information for failing tasks"""
    
    # Failing task IDs identified from previous analysis
    failing_tasks = {
        'telecom': [
            '[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:Hard]',
            '[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]',
            '[mms_issue]bad_network_preference|break_app_sms_permission|user_abroad_roaming_disabled_on[PERSONA:Hard]',
            '[mms_issue]bad_network_preference|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:None]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]',
            '[mms_issue]bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]',
            '[mms_issue]bad_wifi_calling|break_apn_mms_setting|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]',
            '[mms_issue]airplane_mode_on|break_app_both_permissions|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]',
            '[mms_issue]bad_network_preference|bad_wifi_calling|break_app_both_permissions|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Hard]',
            '[mms_issue]bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]',
            '[mms_issue]bad_network_preference|break_app_sms_permission|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]',
            '[mms_issue]bad_network_preference|bad_wifi_calling|break_app_both_permissions|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:None]',
            '[mms_issue]airplane_mode_on|bad_network_preference|break_app_both_permissions|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]',
            '[mms_issue]bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:None]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]',
            '[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]'
        ],
        'airline': ['7', '14', '32', '37', '23', '35'],
        'retail': []  # No failing tasks found
    }
    
    # Get all JSON files
    json_files = glob.glob("*.json")
    
    # Group files by domain
    domain_files = defaultdict(list)
    
    for filepath in json_files:
        if filepath.endswith('.json') and not filepath.endswith('convert.py') and not filepath.endswith('analyze_files.py'):
            data = load_json_file(filepath)
            if data:
                domain = data.get('info', {}).get('environment_info', {}).get('domain_name', 'unknown')
                domain_files[domain].append({
                    'filepath': filepath,
                    'data': data
                })
    
    # Extract task details for each domain
    for domain in ['telecom', 'airline', 'retail']:
        print(f"\n{'='*80}")
        print(f"DETAILED TASK INFORMATION FOR {domain.upper()} DOMAIN")
        print(f"{'='*80}")
        
        if domain not in domain_files:
            print(f"No files found for {domain} domain")
            continue
        
        if not failing_tasks[domain]:
            print(f"No failing tasks found for {domain} domain")
            continue
        
        # Use the first file to get task information
        first_file = domain_files[domain][0]
        data = first_file['data']
        
        print(f"\nFailing tasks for {domain} domain:")
        print("-" * 60)
        
        for task_id in failing_tasks[domain]:
            print(f"\nTask ID: {task_id}")
            print("-" * 40)
            
            # Find task information
            task_info = None
            for task in data.get('tasks', []):
                if task.get('task_id') == task_id:
                    task_info = task
                    break
            
            if task_info:
                print(f"Description: {task_info.get('description', 'N/A')}")
                print(f"Task Type: {task_info.get('task_type', 'N/A')}")
                print(f"Difficulty: {task_info.get('difficulty', 'N/A')}")
                
                # Print initial state
                initial_state = task_info.get('initial_state', {})
                if initial_state:
                    print(f"Initial State:")
                    for key, value in initial_state.items():
                        print(f"  {key}: {value}")
                
                # Print goal state if available
                goal_state = task_info.get('goal_state', {})
                if goal_state:
                    print(f"Goal State:")
                    for key, value in goal_state.items():
                        print(f"  {key}: {value}")
                
                # Print success criteria if available
                success_criteria = task_info.get('success_criteria', {})
                if success_criteria:
                    print(f"Success Criteria:")
                    for key, value in success_criteria.items():
                        print(f"  {key}: {value}")
                
                # Print any additional task metadata
                for key, value in task_info.items():
                    if key not in ['task_id', 'description', 'task_type', 'difficulty', 'initial_state', 'goal_state', 'success_criteria']:
                        print(f"{key}: {value}")
            else:
                print(f"Task information not found for task ID: {task_id}")
            
            print("-" * 60)

if __name__ == "__main__":
    extract_task_details()
