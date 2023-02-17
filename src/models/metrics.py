from keras import backend as K


def r2_score(y_true, y_pred):
    """
    Coefficient of determination
    :param y_true:
    :param y_pred:
    :return: r2_score (-infinity,1]
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def r2_score_pred(y_true, y_pred):
    """
    Coefficient of determination
    :param y_true:
    :param y_pred:
    :return: r2_score (-infinity,1]
    """
    y_true, y_pred = K.constant(y_true), K.constant(y_pred)
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())
