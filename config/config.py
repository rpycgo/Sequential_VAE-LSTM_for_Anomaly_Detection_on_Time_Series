class model_config:
    batch_size=32
    D0 = 3.1
    learning_rate = 1e-2
    h_t = 24
    k = 2

    latent_dim = 2
    time_sequence= 30

    regularization = {
        'beta': 3,
        'lambda': 10
    }

    encoder = {        
        'fc1_units': 128,
        'fc2_units': 64,
        'fc3_units': 32,
        }
    decoder = {
        'fc1_units': 32,
        'fc2_units': 64,
        'fc3_units': 128,
        }
