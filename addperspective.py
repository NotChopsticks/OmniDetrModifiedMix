import json

def add_perspective_field(json_content, perspective_value):
    annotations = json_content.get('annotations', [])
    
    for annotation in annotations:
        annotation['perspective'] = perspective_value
    
    return json_content

# Load your JSON file
with open('D:/General_Storage/SDU-GAMODv4/SDU-GAMODv4/supervised_annotations/ground/aligned_ids/ground_train_aligned_ids_w_indicator.json', 'r') as file:
    json_content = json.load(file)

# Add the "perspective" field with the desired value
new_json_content = add_perspective_field(json_content, 'ground')

# Save the modified JSON content back to the file (or a new file)
with open('D:/General_Storage/SDU-GAMODv4/SDU-GAMODv4/supervised_annotations/ground/aligned_ids/ground_train_aligned_ids_w_perspective.json', 'w') as file:
    json.dump(new_json_content, file, indent=6)
