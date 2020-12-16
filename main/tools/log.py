def log(i, filename, message):
    """
    :param i: i=-1 -> ERROR ; i=0 -> WARN ; i=1 -> INFO
    :param filename:
    :param message:
    :return: a log message
    """
    if i == -1:
        return "[ERROR] {}: {}".format(filename, message)
    elif i == 0:
        return "[WARN] {}: {}".format(filename, message)
    elif i == 1:
        return "[INFO] {}: {}".format(filename, message)
