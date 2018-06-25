from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest, f_classif

def select_k_best(data_dict, feature_list, features_qty):

    data = featureFormat(data_dict, feature_list)
    labels, features = targetFeatureSplit(data)

    clf = SelectKBest(f_classif, k=features_qty)
    clf.fit(features, labels)

    result = zip(feature_list[1:], clf.scores_)
    result = sorted(result, key=lambda item: item[1], reverse=True)

    return result[:features_qty]

def create_poi_email_ratio_features(data_dict):
    """
    Cria duas novas features com a proporcao de emails recebidos e enviados a POI em relacao ao total de emails
    :param data_dict: Dicionario de dados com as features existentes
    :return: data_dict: Dicionario de dados atualizado com a nova feature
             feature_name: Array com nome das novas features
    """
    new_features = []

    data_dict, feature = create_poi_email_to_ratio(data_dict)
    new_features.append(feature)
    data_dict, feature = create_poi_email_from_ratio(data_dict)
    new_features.append(feature)

    return data_dict, new_features

def create_new_financial_features(data_dict):
    """
    Cria duas novas features com base no bonus, total de pagamentos e total de acoes
    :param data_dict: Dicionario de dados com as features existentes
    :return: data_dict: Dicionario de dados atualizado com a nova feature
             feature_name: Array com nome das novas features
    """
    new_features = []

    data_dict, feature = create_total_revenue_feature(data_dict)
    new_features.append(feature)
    data_dict, feature = create_bonus_ratio_on_total_payments(data_dict)
    new_features.append(feature)

    return data_dict, new_features



def create_total_revenue_feature(data_dict):
    """
    Cria uma nova feature com a soma do total dos pagamentos e tota de acoes doe cada registro
    :param data_dict: Dicionario de dados com as features existentes
    :return: data_dict: Dicionario de dados atualizado com a nova feature
             feature_name: Nome da nova features
    """
    feature_name = 'total_revenue'
    for data in data_dict:
        if data_dict[data]['total_payments'] == "NaN" or data_dict[data]['total_payments'] == 0:
            data_dict[data][feature_name] = data_dict[data]['total_stock_value']
        elif data_dict[data]['total_stock_value'] == "NaN" or data_dict[data]['total_stock_value'] == 0:
            data_dict[data][feature_name] = data_dict[data]['total_payments']
        else:
            data_dict[data][feature_name] = data_dict[data]['total_payments'] + data_dict[data]['total_stock_value']

    return data_dict, feature_name

def create_bonus_ratio_on_total_payments(data_dict):
    """
    Cria uma nova feature com a proporcao do bonus sobre o total de pagaments recebidos
    :param data_dict: Dicionario de dados com as features existentes
    :return: data_dict: Dicionario de dados atualizado com a nova feature
             feature_name: Nome da nova features
    """
    feature_name = 'bonus_ratio'
    for data in data_dict:
        if data_dict[data]['bonus'] == "NaN" or \
                data_dict[data]['total_payments'] == "NaN" or \
                data_dict[data]['bonus'] == 0 or \
                data_dict[data]['total_payments'] == 0:
            data_dict[data][feature_name] = 0
        else:
            data_dict[data][feature_name] = data_dict[data]['bonus'] / data_dict[data]['total_payments']

    return data_dict, feature_name


def create_poi_email_to_ratio(data_dict):
    """
    Cria duas novas nova feature com a soma do total dos pagamentos e tota de acoes doe cada registro
    :param data_dict: Dicionario de dados com as features existentes
    :return: data_dict: Dicionario de dados atualizado com a nova feature
             feature_name: Nome da nova features
    """
    feature_name = 'ratio_email_to_poi'
    for data in data_dict:
        if data_dict[data]['from_poi_to_this_person'] == "NaN" or \
                data_dict[data]['to_messages'] == "NaN" or \
                data_dict[data]['from_poi_to_this_person'] == 0 or \
                data_dict[data]['to_messages'] == 0:
            data_dict[data][feature_name] = 0
        else:
            data_dict[data][feature_name] = data_dict[data]['from_poi_to_this_person'] / data_dict[data]['to_messages']

    return data_dict, feature_name

def create_poi_email_from_ratio(data_dict):
    """
    Cria duas novas nova feature com a soma do total dos pagamentos e tota de acoes doe cada registro
    :param data_dict: Dicionario de dados com as features existentes
    :return: data_dict: Dicionario de dados atualizado com a nova feature
             feature_name: Nome da nova features
    """
    feature_name = 'ratio_email_from_poi'
    for data in data_dict:
        if data_dict[data]['from_this_person_to_poi'] == "NaN" or \
                data_dict[data]['from_messages'] == "NaN" or \
                data_dict[data]['from_this_person_to_poi'] == 0 or \
                data_dict[data]['from_messages'] == 0:
            data_dict[data][feature_name] = 0
        else:
            data_dict[data][feature_name] = data_dict[data]['from_this_person_to_poi'] / data_dict[data]['from_messages']

    return data_dict, feature_name
