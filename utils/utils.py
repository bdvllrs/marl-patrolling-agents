def compute_discounted_return(gamma, rewards):
    discounted_return = 0
    discount = 1
    for reward in rewards:
        discounted_return += reward * discount
        discount *= gamma
    return discounted_return
