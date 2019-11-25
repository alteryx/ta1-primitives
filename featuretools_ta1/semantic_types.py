import featuretools as ft

PRIMARY_KEY = "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
TEXT = "http://schema.org/Text"
NUMBER = "http://schema.org/Number"
INTEGER = "http://schema.org/Integer"
FLOAT = "http://schema.org/Float"
DATETIME = "http://schema.org/DateTime"
BOOLEAN = "http://schema.org/Boolean"
CATEGORICAL = "https://metadata.datadrivendiscovery.org/types/CategoricalData"
ORDINAL = "https://metadata.datadrivendiscovery.org/types/OrdinalData"
TARGET = "https://metadata.datadrivendiscovery.org/types/Target"
TRUE_TARGET = "https://metadata.datadrivendiscovery.org/types/TrueTarget"
ATTRIBUTE = "https://metadata.datadrivendiscovery.org/types/Attribute"


ft_to_d3m = {
    ft.variable_types.Text: TEXT,
    ft.variable_types.Numeric: NUMBER,
    ft.variable_types.Datetime: DATETIME,
    ft.variable_types.Categorical: CATEGORICAL,
    ft.variable_types.Boolean: BOOLEAN,
    ft.variable_types.Ordinal: ORDINAL,
    ft.variable_types.Index: PRIMARY_KEY,
    ft.variable_types.Id: CATEGORICAL,  # best matching
}

d3m_to_ft = {
    BOOLEAN: ft.variable_types.Boolean,
    TEXT: ft.variable_types.Text,
    NUMBER: ft.variable_types.Numeric,
    FLOAT: ft.variable_types.Numeric,
    INTEGER: ft.variable_types.Numeric,
    DATETIME: ft.variable_types.Datetime,
    CATEGORICAL: ft.variable_types.Categorical,
    ORDINAL: ft.variable_types.Ordinal,
    PRIMARY_KEY: ft.variable_types.Index,
}
