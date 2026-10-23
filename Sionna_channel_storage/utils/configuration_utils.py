def group_configurations(configurations):
    """Group configurations that share centered subarray coordinates."""
    groups = {}
    for cfg in configurations:
        group_key = (
            cfg["fc"],
            cfg["tx_rows"] % 2,
            cfg["tx_cols"] % 2,
            cfg["rx_rows"] % 2,
            cfg["rx_cols"] % 2,
        )
        group = groups.get(group_key)
        if group is None:
            group = {
                "fc": cfg["fc"],
                "tx_rows": cfg["tx_rows"],
                "tx_cols": cfg["tx_cols"],
                "rx_rows": cfg["rx_rows"],
                "rx_cols": cfg["rx_cols"],
                "configurations": [],
            }
            groups[group_key] = group
        else:
            group["tx_rows"] = max(group["tx_rows"], cfg["tx_rows"])
            group["tx_cols"] = max(group["tx_cols"], cfg["tx_cols"])
            group["rx_rows"] = max(group["rx_rows"], cfg["rx_rows"])
            group["rx_cols"] = max(group["rx_cols"], cfg["rx_cols"])
        group["configurations"].append(cfg)

    configuration_groups = list(groups.values())
    return configuration_groups


def centered_array_indices(full_rows, full_cols, rows, cols):
    """Return flattened PlanarArray indices for a centered subarray."""
    
    row_start = (full_rows - rows) // 2
    col_start = (full_cols - cols) // 2
    indices = []
    for col in range(col_start, col_start + cols):
        start = row_start + col * full_rows
        indices.extend(range(start, start + rows))

    return indices
