import json


def tiers(hydrated_data):
    source_index = hydrated_data["node_mapping"][hydrated_data["thread_id"]]
    edges = hydrated_data["edges"]

    first_level_replies = []
    other_level_replies = []
    
    # Find first level replies
    for edge in edges:
        if source_index in edge: 
            if edge[0]!=source_index:
                first_level_replies.append(edge[0])
            else:
                first_level_replies.append(edge[1]) 

    # Find other level replies
    for edge in edges:
        if source_index not in edge:
            if edge[0] not in first_level_replies:
                other_level_replies.append(edge[0])
            if edge[1] not in first_level_replies:
                other_level_replies.append(edge[1])  

    return source_index, first_level_replies,other_level_replies              


