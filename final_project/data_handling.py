def outlier_cleaning(data):
    """
    Responsavel pela limpeza dos outliers do conjunto de dados da Enron.
    :param data: Dicionario de dados para limpeza
    :return: Dados com outliers removidos

    """

    outlier_keys = ['LOCKHART EUGENE E',
                    'THE TRAVEL AGENCY IN THE PARK',
                    'TOTAL']

    for key in outlier_keys:
        data.pop(key)

    return data




def data_handling(data):
    """
    Responsavel pela correcao das informacoes no conjunto de dados da Enron.
    :param data: Dicionario de dados para correcao
    :return: Dados com dados corrigidos
    """
    ### Ajusta dos valores de BHATNAGAR SANJAY
    data['BHATNAGAR SANJAY']['total_stock_value'] = data['BHATNAGAR SANJAY']['restricted_stock_deferred']
    data['BHATNAGAR SANJAY']['restricted_stock_deferred'] = data['BHATNAGAR SANJAY']['restricted_stock']
    data['BHATNAGAR SANJAY']['restricted_stock'] = data['BHATNAGAR SANJAY']['exercised_stock_options']
    data['BHATNAGAR SANJAY']['exercised_stock_options'] = data['BHATNAGAR SANJAY']['total_payments']
    data['BHATNAGAR SANJAY']['total_payments'] = data['BHATNAGAR SANJAY']['director_fees']
    data['BHATNAGAR SANJAY']['director_fees'] = data['BHATNAGAR SANJAY']['expenses']
    data['BHATNAGAR SANJAY']['expenses'] = data['BHATNAGAR SANJAY']['other']
    data['BHATNAGAR SANJAY']['other'] = 0

    ### Ajuste dos valores de BELFER ROBERT
    data['BELFER ROBERT']['deferred_income'] = data['BELFER ROBERT']['deferral_payments']
    data['BELFER ROBERT']['deferral_payments'] = data['BELFER ROBERT']['expenses']
    data['BELFER ROBERT']['loan_advances'] = data['BELFER ROBERT']['other']
    data['BELFER ROBERT']['other'] = data['BELFER ROBERT']['expenses']
    data['BELFER ROBERT']['expenses'] = data['BELFER ROBERT']['director_fees']
    data['BELFER ROBERT']['director_fees'] = data['BELFER ROBERT']['total_payments']
    data['BELFER ROBERT']['total_payments'] = data['BELFER ROBERT']['exercised_stock_options']
    data['BELFER ROBERT']['exercised_stock_options'] = data['BELFER ROBERT']['restricted_stock']
    data['BELFER ROBERT']['restricted_stock'] = data['BELFER ROBERT']['restricted_stock_deferred']
    data['BELFER ROBERT']['restricted_stock_deferred'] = data['BELFER ROBERT']['total_stock_value']
    data['BELFER ROBERT']['total_stock_value'] = 0

    return data


def remove_nan(data):
    for linha in data:
        for campo in data[linha]:
            if data[linha][campo] == "NaN":
                data[linha][campo] = 0
    return data