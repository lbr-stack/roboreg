from setuptools import setup

setup(
    name="roboreg",
    version="1.0.0",
    author="mhubii",
    author_email="m.huber_1994@hotmail.de",
    description="Unified eye-in-hand / eye-to-hand calibration from RGB-D images using robot mesh as calibration target.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        "faiss-gpu",
        "huggingface_hub",
        "kinpy",
        "matplotlib",
        "ninja",
        "numpy",
        "open3d",
        "opencv-python",
        "pytorch_kinematics",
        "rich",
        "torch",
        "trimesh",
        "xacro",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CC BY-NC 4.0 Deed",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/lbr-stack/roboreg",
    project_urls={
        "Homepage": "https://github.com/lbr-stack/roboreg",
        "Issues": "https://github.com/lbr-stack/roboreg/issues",
    },
    entry_points={
        "console_scripts": [
            "rr-stereo-dr=roboreg.cli.rr_stereo_dr:main",
            "rr-hydra=roboreg.cli.rr_hydra:main",
            "rr-render=roboreg.cli.rr_render:main",
            "rr-sam=roboreg.cli.rr_sam:main",
            "rr-sam2=roboreg.cli.rr_sam2:main",
        ]
    },
)
