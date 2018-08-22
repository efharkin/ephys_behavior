from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = []
    for line in f.read().splitlines():
        if not line.startswith("--"):
            requirements.append(line)

packages = find_packages()

setup(name="sync",
      version="1.3.1",
      description="AIBS Sync Package",
      author="derricw",
      author_email="derricw@alleninstitute.org",
      url="http://stash.corp.alleninstitute.org/projects/ENG/repos/sync/browse",
      packages=packages,
      install_requires=requirements,
      dependency_links=["http://aibspi:8000/toolbox/"],
      include_package_data=True,
      package_data={
          "": ['*.png', '*.ico', '*.jpg', '*.jpeg'],
          },
      )
