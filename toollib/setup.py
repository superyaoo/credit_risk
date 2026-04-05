from setuptools import setup, find_packages

requires = ["numpy>=1.26.4",
            "pandas>=2.2.2", ]

setup(
    name="toollib",
    version="2.0",
    author="YuChuang risk models",
    description="model utils",
    packages=find_packages(),
    install_requires=requires,
    python_requires='>=3.10',
    include_package_data=True,
    package_data={
        'toollib': ['model_report/model_report_template.xlsx',
                    'model_report/feature_report_template.xlsx',
                    'model_report/model_report_template_v2.xlsx']
    }
)
