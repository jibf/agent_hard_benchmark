import argparse
import re

def calculate_average(file_path):
    values = []
    pattern = re.compile(r':\s*([\d.]+)$')
    
    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                match = pattern.search(line)
                if match:
                    try:
                        value = float(match.group(1))
                        values.append(value)
                    except ValueError:
                        print(f"Warning: Invalid number format at line {line_number}: '{line}'")
                else:
                    print(f"Warning: No colon+number pattern found at line {line_number}: '{line}'")
        
        if not values:
            print("Error: No valid values found in the file")
            return None
        
        total = sum(values)
        count = len(values)
        average = total / count
        return total, count, average
    
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average of values after colons in a text file')
    parser.add_argument('--log', type=str, required=True, help='Path to the text file')
    args = parser.parse_args()
    
    result = calculate_average(args.log)
    
    if result is not None:
        total, count, average = result
        print(f"Values extracted: {count}")
        print(f"Total sum: {total:.6f}")
        print(f"Average: {average:.6f}")