# reconstruction slices defined for different datasets at different fps
# (dataset_name, fps): {"recon_slice": (slice(start, end), slice(None), slice(None))}
data_configs = {
    # new captures
    ("plasma-ball", 50000): {
        "recon_slice": (slice(20000, 25000), slice(None), slice(None)),
    },
    ("bright1", 100_000): {
        "recon_slice": (slice(11300, 15500), slice(None), slice(None)),
    },
    ("bright2", 100_000): {
        "recon_slice": (slice(2400, 6100), slice(None), slice(None)),
    },
    ("dark1", 100_000): {
        "recon_slice": (slice(36500, 41700), slice(None), slice(None)),
    },
    ("ultra_dark1", 100_000): {
        "recon_slice": (slice(36400, 41400), slice(None), slice(None)),
    },
    ("ultra_dark2", 100_000): {
        "recon_slice": (slice(0, 5000), slice(None), slice(None)),
    },
    ("fanclock_bright", 100_000): {
        "recon_slice": (slice(1000, 3000), slice(None), slice(None)),
    },
    ("fanclock_bright_spadnd", 100_000): {
        "recon_slice": (slice(1000, 3000), slice(None), slice(None)),
    },
    ("fanclock_dark", 100_000): {
        "recon_slice": (slice(1000, 3000), slice(None), slice(None)),
    },
    ("teaser-gunballoon-dark-acq00002", 100_000): {
        "recon_slice": (slice(83500, 89000), slice(None), slice(None)),
    },
    ("teaser_balloonbounce_dark", 100_000): {
        "recon_slice": (slice(24300, 37900), slice(None), slice(None)),
    },
    ("teaser_balloonbounce_bright", 100_000): {
        "recon_slice": (slice(33600, 46500), slice(None), slice(None)),
        # "recon_slice": (slice(None, None, 500), slice(None), slice(None)),
    },
    ("teaser-blender-dark", 100_000): {
        "recon_slice": (slice(70000, 73000), slice(None), slice(None)),
    },
    ("teaser-blender-bright1", 100_000): {
        "recon_slice": (slice(39000, 42000), slice(None), slice(None)),
    },
    ("balloon-laser-acq00000", 100_000): {
        "recon_slice": (slice(5000, 15000), slice(None), slice(None)),
    },
    ("balloon-laser-acq00001", 100_000): {
        "recon_slice": (slice(30000, 40000), slice(None), slice(None)),
    },
    ("balloon-laser-acq00002", 100_000): {
        "recon_slice": (slice(30000, 40000), slice(None), slice(None)),
    },
    ("Feb27_balloonbounce_acq3_2740ppps", 100_000): {
        "recon_slice": (slice(12800, 24200), slice(None), slice(None)),
    },
    ("Feb27_balloonbounce_acq4_3100ppps", 100_000): {
        "recon_slice": (slice(23000, 33000), slice(None), slice(None)),
        # "recon_slice": (slice(28000, 33000), slice(None), slice(None)),
    },
    ("Feb27_balloonbounce_acq7_1000ppps", 100_000): {
        "recon_slice": (slice(18500, 30000), slice(None), slice(None)),
    },
    ("Feb28_balloonBounce_acq0_2221ppps", 100_000): {
        # focused window: 23200, 23900
        "recon_slice": (slice(19600, 29500), slice(None), slice(None)),
    },
    ("Feb28_balloonBounce_acq0_2221ppps", 10_000): {
        "recon_slice": (slice(1960, 2950), slice(None), slice(None)),
    },
    ("Feb28_balloonBounce_acq0_2221ppps", 2_000): {
        "recon_slice": (slice(392, 590), slice(None), slice(None)),
    },
    ("Feb28_balloonBounce_acq2_2365ppps", 100_000): {
        "recon_slice": (slice(33300, 42500), slice(None), slice(None)),
    },
    ("Feb29_balloonbounce_acq0_2875ppps", 100_000): {
        "recon_slice": (slice(18500, 28100), slice(None), slice(None)),
    },
    ("Feb27_balls_acq9_2948", 100_000): {
        "recon_slice": (slice(59100, 76000), slice(None), slice(None)),
    },
    ("Feb27_balls_acq11_5114", 100_000): {
        "recon_slice": (slice(50600, 68000), slice(None), slice(None)),
    },
    ("Feb27_blender_acq13_6041ppps", 100_000): {
        "recon_slice": (slice(90000, 100000), slice(None), slice(None)),
    },
    ("Feb27_blender_acq15_4351ppps", 100_000): {
        "recon_slice": (slice(65000, 75000), slice(None), slice(None)),
    },
    ("Feb28_blender_acq7_2986ppps", 100_000): {
        "recon_slice": (slice(50000, 58000), slice(None), slice(None)),
    },
    # old quanta data
    ("fan", 100_000): {
        "recon_slice": (slice(70000, 72000), slice(None), slice(None)),
    },
    # qbp data
    ("0116-guitar-f4-tworow-aa-3", 100_000): {
        "recon_slice": (slice(42500, 48700), slice(None), slice(None)),
    },
    # bit2bit data
    ("Monkey", 100_000): {
        "recon_slice": (slice(0, 10000), slice(None), slice(None)),
    },
    ("Resolution_target_drill", 100_000): {
        "recon_slice": (slice(0, 3000), slice(None), slice(None)),
    },
    ("cpufan_restarget", 100_000): {
        "recon_slice": (slice(20000, 25000), slice(None), slice(None)),
    },
    ("plasma_ball_5_med", 100_000): {
        "recon_slice": (slice(20000, 25000), slice(None), slice(None)),
    },
    ("ultrasound_bubble24", 100_000): {
        "recon_slice": (slice(15000, 25000, 50), slice(None), slice(None)),
    },
    # misc data
    ("blender_lasagna_bright", 50000): {
        "recon_slice": (slice(30000, 32000), slice(None), slice(None)),
    },
    ("blender_lasagna_dark", 50000): {
        "recon_slice": (slice(30000, 32000), slice(None), slice(None)),
    },
    ("blender_lasagna_dark", 100000): {
        "recon_slice": (slice(25000, 27000), slice(None), slice(None)),
    },
    ("balloon1", 100_000): {
        "recon_slice": (slice(71800, 73800), slice(None), slice(None)),
    },
    ("balloon_gun1", 50000): {
        "recon_slice": (slice(38350, 40350), slice(None), slice(None)),
    },
    ("balloon_gun2", 50000): {
        "recon_slice": (slice(42500, 44500), slice(None), slice(None)),
    },
    ("balloon_gun3", 50000): {
        "recon_slice": (slice(0, 2000), slice(None), slice(None)),
    },
    ("balloon_gun3", 100000): {
        "recon_slice": (slice(2000, 4000), slice(None), slice(None)),
    },
    ("gun_balloon_dark", 50000): {
        "recon_slice": (slice(12000, 14000), slice(None), slice(None)),
    },
    ("balloon_fly1", 50000): {
        "recon_slice": (slice(2350, 4350), slice(None), slice(None)),
    },
    ("balloon_fly1_alt", 50000): {
        "recon_slice": (slice(6000, 8000), slice(None), slice(None)),
    },
    ("balloon_fly2", 50000): {
        "recon_slice": (slice(4750, 6750), slice(None), slice(None)),
    },
    ("lasagna2", 50000): {
        "recon_slice": (slice(18939, 20939), slice(None), slice(None)),
    },
}


WINDOW_SETTINGS_DEFAULT = {
    "chunk_t": 10000,
    "chunk_h": 100,
    "chunk_w": 100,
    "hop_t": 2500,
    "hop_h": 25,
    "hop_w": 25
}
WINDOW_SETTINGS_BRIGHT = {
    "chunk_t": 5000,
    "chunk_h": 64,
    "chunk_w": 64,
    "hop_t": 1250,
    "hop_h": 16,
    "hop_w": 16
}
# about as much as I can stuff in VRAM
WINDOW_SETTINGS_DARK = {
    "chunk_t": 8000,
    "chunk_h": 100,
    "chunk_w": 100,
    "hop_t": 2000,
    "hop_h": 25,
    "hop_w": 25
}
WINDOW_SETTINGS_SPEED = {
    "chunk_t": 5000,
    "chunk_h": 64,
    "chunk_w": 64,
    "hop_t": 2500,
    "hop_h": 32,
    "hop_w": 32
}
