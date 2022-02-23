import __main__
from scipy.sparse import csr_matrix
from collections import Counter
import numpy as np

ids = None


class IdTransform:
    def __init__(self):
        userids = __main__.user_item_matrix.index.values
        itemids = __main__.user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))

        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))
        self.sparce_user_item_matrix = csr_matrix(__main__.user_item_matrix).tocsr()


def ids_stored():
    if ids is None:
        return False
    else:
        return True


def get_recommendations(user, model, N=5):
    global ids
    if not ids_stored():
        ids = IdTransform()
    res = [ids.id_to_itemid[rec[0]] for rec in
           model.recommend(userid=ids.userid_to_id[user],
                           user_items=ids.sparce_user_item_matrix,  # на вход user-item matrix
                           N=N,
                           filter_already_liked_items=False,
                           filter_items=[ids.itemid_to_id[999999]],
                           recalculate_user=True)]
    return res


def get_top_items(user, N=5):
    user_top_n = __main__.user_item_matrix.loc[user].sort_values(ascending=False)
    user_top_n = user_top_n.head(N).index
    return user_top_n


def get_similar_items_recommendations(user, model, N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    global ids
    if not ids_stored():
        ids = IdTransform()
    user_top_n = get_top_items(user, N=N)
    recs = []
    for item in user_top_n:
        recs.extend([ids.id_to_itemid[i[0]] for i in model.similar_items(ids.itemid_to_id[item], 5)[1:]])
    recs = Counter(recs)
    if 999999 in recs:
        del recs[999999]
    res = [i[0] for i in recs.most_common(N)]
    return res


def get_similar_users_recommendations(user, model, N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    global ids
    if not ids_stored():
        ids = IdTransform()
    sim_users = [u[0] for u in model.similar_users(ids.userid_to_id[user], N=5)]
    recs = []
    for u in sim_users:
        recs.extend(get_top_items(ids.id_to_userid[u], N=5))
    recs = Counter(recs)
    if 999999 in recs:
        del recs[999999]
    res = [i[0] for i in recs.most_common(N)]
    return res
