def mask(values, users, gamma):
    # print(values,users,gamma)
    values_gamma = values[:, gamma == users]
    # print(values_gamma)
    return values_gamma