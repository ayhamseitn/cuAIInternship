import json

def main():
    # Load the data file
    data_file = input("Please enter the path to your CoQA data file: ")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Print the JSON structure
    print(json.dumps(data, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
