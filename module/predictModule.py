def assign_node_predict_label(nodeInfo, threshold):
    cost_benign = (1 - nodeInfo['prior_probability'][0]) + nodeInfo['msg_sum'][0]
    cost_phish = (1 - nodeInfo['prior_probability'][1]) + nodeInfo['msg_sum'][1]
    normalized_probability_benign, normalized_probability_phish = normalize_cost(cost_benign, cost_phish)
    if normalized_probability_benign >= threshold:
        return 0
    else:
        return 1


def normalize_cost(cost_benign, cost_phish):
    total_cost = cost_benign + cost_phish
    if total_cost != 0:
        normalized_probability_benign = 1 - cost_benign / total_cost
        normalized_probability_phish = 1 - cost_phish / total_cost
    else:
        normalized_probability_benign = 0.5
        normalized_probability_phish = 0.5
    return normalized_probability_benign, normalized_probability_phish

