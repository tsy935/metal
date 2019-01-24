from metal.utils import SlicingDataset


def train_models(
    config,
    Ls,
    Xs,
    Ys,
    Zs,
    verbose=False):
    """
    Generates weak labels and trains a single model
    Returns:
        model: a trained model
    """
    assert(isinstance(config["L_weights"], list) or config["L_weights"] is None)

    # Generate weak labels:
    # a) uniform (L_weights = [1,...,1])
    # b) manual  (L_weights = [1,X,...1])
    # c) learned (L_weights = None): DP
    Y_weak = generate_weak_labels(Ls[0], config["L_weights"])

    # Instantiate end model
    model = EndModel(layer_output)

    # Add slice hat if applicable
    slice_kwargs = config.get('slice_kwargs')
    if slice_kwargs:
        model = SliceHatModel(model, m, **slice_kwargs)


    base_model_class = config["base_model_class"]
    base_model_init_kwargs = config["base_model_init_kwargs"]
    train_kwargs = config["train_kwargs"]

    model = base_model_class(
#            input_module=input_module_class(**input_module_init_kwargs),
        **base_model_init_kwargs,
        verbose=verbose,
        use_cuda=use_cuda,
        seed=seed,
    )
    # Make dataloaders



    train_dataset = (
        SlicingDataset(Xs[0], Y_weak, Ls[0]) if config["train_on_L"]
        else SyntheticDataset(Xs[0], Y_weak)
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=train_kwargs.get("batch_size", 32),
        num_workers=1,
        sampler=sampler
    )




#        input_module_class = config["input_module_class"]
#        input_module_init_kwargs = config["input_module_init_kwargs"]

    # init base model (i.e. EndModel or SliceDPModel)



        # train model
        model.train_model(
            train_loader,
            dev_data=(Xs[1], Ys[1]),
            **train_kwargs,
        )

        # collect trained models in dict
        trained_models[model_name] = model

    return trained_models


def eval_model(
    model,
    eval_loader,
    metrics=["accuracy", "precision", "recall", "f1"],
    verbose=True,
    break_ties="random",
):
    """
    Args:
        model: a trained EndModel (or subclass)
        eval_loader: a loader containing X, Y, Z
    """
    X, Y, Z = separate_eval_loader(eval_loader)
    out_dict = {}

    # Evaluating on full dataset
    if verbose:
        print(f"All: {len(Z)} examples")
    metrics_full = model.score((X, Y), metrics, verbose=verbose)
    out_dict["all"] = {metrics[i]: metrics_full[i] for i in range(len(metrics))}

    # Evaluating on slice
    slices = sorted(set(Z))
    for s in slices:

        # Getting indices of points in slice
        inds = [i for i, e in enumerate(Z) if e == s]
        if verbose:
            print(f"\nSlice {s}: {len(inds)} examples")
        X_slice = X[inds]
        Y_slice = Y[inds]

        metrics_slice = model.score(
            (X_slice, Y_slice), metrics, verbose=verbose, break_ties=break_ties
        )

        out_dict[f"slice_{s}"] = {
            metrics[i]: metrics_slice[i] for i in range(len(metrics_slice))
        }

    print("\nSUMMARY (accuracies):")
    print(f"All: {out_dict['all']['accuracy']}")
    for s in slices:
        print(f"Slice {s}: {out_dict['slice_' + s]['accuracy']}")

    return out_dict


def separate_eval_loader(data_loader):
    X = []
    Y = []
    Z = []

    # The user passes in a single data_loader and we handle splitting and
    # recombining
    for ii, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        x_batch, y_batch, z_batch = data

        X.append(x_batch)
        Y.append(y_batch)
        if isinstance(z_batch, torch.Tensor):
            z_batch = z_batch.numpy()
        Z.extend([str(z) for z in z_batch])  # slice labels may be strings

    X = torch.cat(X)
    Y = torch.cat(Y)
    return X, Y, Z
