from recommendation.recommender import ProductRecommender

_recommender = None


def get_recommender():
    global _recommender

    if _recommender is None:
        _recommender = ProductRecommender()

    return _recommender


def get_recommendations(customer_id, top_n=10):
    recommender = get_recommender()
    return recommender.recommend(customer_id, top_n)