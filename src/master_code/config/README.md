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

`parser.py` contains the main configuration dataclass `SlamConfig` which is used throughout the program. This dataclass contains the dataclasses defined in `config.py` as fields. `parser.py` also supplies the `SlamConfig.load()` class method and the `SlamConfig.save()` instance method, which are used to load and save the configuration from and to a specified YAML file.


`files/` is the folder in which all configuration YAML files are stored. `default_config.yaml` corresponds to the default values defined in `SlamConfig` and can be created by calling `SlamConfig().save("default_config.yaml")`.

# Usage

The default values of the dataclasses in `config.py` should **NOT** be changed directly, as they act as the default experimentation values. Instead, create a copy of the base config YAML file, change the values appropriately, and then create a `SlamConfig` instance by calling `SlamConfig.load(...)` on the appropriate file. *OmegaConf* ensures that the values specified in the YAML file overwrite the default values of `SlamConfig`. This also means that not all fields need to be defined in the YAML file, since undefined fields will keep the default values defined in `SlamConfig`.

If you want to change or add new fields to the config, first add the new field with default values to the appropriate dataclass in `config.py`. Then you can either manually update the YAML files with the new fields or call `SlamConfig().save(...)` to create a new default YAML config file with the new fields added. Keep in mind that changing existing field names will deprecate previous YAML files due to name conflicts, while adding new fields will not, since older YAML files will simply use the default values for the new fields.
