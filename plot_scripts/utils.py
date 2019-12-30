import numpy as np

def get_agent_parse_info(agent_json_data, divide_type=None):

    # pre_divider, num_types, post_divider
    pre_divider = 1
    num_type = -1
    num_settings = 1

    found = False
    data = agent_json_data['sweeps']

    if divide_type is not None:
        for key in data:
            current_length = len(data[key])

            if key == divide_type:
                found = True
                num_type = current_length
            elif not found:
                pre_divider *= current_length
            num_settings *= current_length

        type_arr = data[divide_type]
        post_divider = int(num_settings / pre_divider)

    # just return num_settings
    else:
        setting_count_arr = [len(data[key]) for key in data]
        num_settings = np.prod(setting_count_arr)

        type_arr = None
        pre_divider = None
        num_type = None
        post_divider = None

    return type_arr, pre_divider, num_type, post_divider, num_settings
