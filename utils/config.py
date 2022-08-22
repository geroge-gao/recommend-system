#

feat_spec = {
    "inputs":{
        # 定义输入数据顺序
        "movie_id": "int",
        "user_id": "int",
        "gender": "int",
        "age": "int",
        "work": "int",
        "zip": "int",
        "movie_types": "list",
        "movie_hist": "list",

    }, 

    "embed":
    {
        "user_id": {"input_dim": 6041, "output_dim": 16},
        "gender": {"input_dim": 2, "output_dim": 16},
        "age": {"input_dim": 7, "output_dim": 16},
        "work": {"input_dim": 21, "output_dim": 16},
        "zip": {"input_dim": 3439, "output_dim": 16},
        "movie_id": {"input_dim": 3706, "output_dim": 16},
        "movie_types": {"input_dim": 19, "output_dim": 16}
    },

    "list_feat":
    {"movie_hist": 19, 
     "movie_types": 6
    },

    "shared_embed": ["movie_id"],


    "share_from_embed":
    {
        "movie_hist": "movie_id"
    },

    "outputs":{"movie_id": "int"}
}