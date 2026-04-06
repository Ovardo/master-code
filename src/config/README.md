# OmegaConf for YAML config parsing

The handling of configuration files is based [*OmegaConf*](https://omegaconf.readthedocs.io/en/2.3_branch/), a YAML based hierichal configuration library with provides seamles parsing of yaml files into Python dataclasses, offering runtime type safety and enabling value validation and default values by the use of dataclasses. 


# Structure

```text
.
├── config.py
├── parser.py
└── files
	├── default_config.yaml
	└── ...
```


`config.py` contains the sub-dataclasses used for storing the different configuration parameters. These classes also include runtime value validation of parsed values, ensuring that invalid values are detected early. When convenient, helper properties are also added to the dataclasses to compute derived values. 

`parser.py` contains the main configration dataclasss `SLAMConfig` which will be used troughet the program. This dataclass contains metedata fields aswell as the dataclasses defined in `config.py` as subclasses. `parser.py` also supplies the functions `load_config()` and `save_config()`which are used to load/save the  configration from/to the specified yaml file.   


`files/` is the folder in which all the configuration yaml files are stored. `default_config.yaml` is the  configuration file corresponding to the default values defined in the `ConfigSLAM` and can been created by running save_config().

# Usage

The default values of the dataclasses in `config.py` should **NOT** be changed directly as they act as the default experimentation values. Instead, one should create a copy of the base_config.yaml file, change the values appropiatly and then create the `SLAMConfig` dataclass by using `load_config()` on the appropiate file. *OmegaConf* ensure the specificed values in the yaml file overwrites the default values of `SLAMConfig`. This also means that one do not need to have all fields in the yaml file defined, as the the undefined fields will get the default values defined in `SLAMConfig`.

If wanting to change or add new fields to the config, one should first add the new field with default values to appropiate dataclass in `config.py`. Then one can either manually update the yaml files with the new fields or run `save_config()` on a default instance of `SLAMConfig` to create a new default yaml config file with the new fields added. Keep in mind that changing existing field names will deprecate previous yaml files due to name conflicts while adding new fields will not as old yaml files will just use the default values for the new fields.


# TODO: 
Could consider turning load_config and save_config into static memeber funcitons of SLAMConfig dataclass. 

