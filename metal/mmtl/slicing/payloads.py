from metal.mmtl.payload import Payload


def remap_labelsets(payload, labels_to_tasks):
    """ Remaps payload.labels_to_tasks based on specified dictionary. All other
        defaults to `labelset` -> `None`.

    Args:
        labels_to_heads: if specified, remaps (in-place) labelsets to specified
            task heads.

    """
    test_labelsets = payload.labels_to_tasks.keys()
    for label_name in test_labelsets:
        if label_name in labels_to_tasks:
            new_task = labels_to_tasks[label_name]
            payload.retarget_labelset(label_name, new_task)
        else:
            payload.retarget_labelset(label_name, None)
