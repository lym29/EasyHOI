import yaml

def to_np(tensor):
    return tensor.cpu().numpy()


def get_data_from_config(config_file):
    # Load the YAML file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    # Access the `defaults` list
    defaults = config.get("defaults", [])
    # Extract the `data` element from the list of dictionaries
    data = None
    for element in defaults:
        if isinstance(element, dict) and "data" in element:
            data = element["data"]
            break  # Exit the loop once we find the `data` key
    return data
