from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from inspect import isclass
from multiprocessing.sharedctypes import Value
from types import GenericAlias, UnionType
from typing import Any, Mapping, Type

import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _assert_type(val, typ):
    if not isinstance(val, typ):
        raise ValueError(f"{val} is not of type {typ}")


def parse(typ, value):
    if is_dataclass(typ):
        return parse_data_class(typ, value)
    elif isinstance(typ, GenericAlias):
        return parse_generic_alias(typ, value)
    elif isinstance(typ, UnionType):
        return parse_union_type(typ, value)
    elif not isclass(typ):
        raise ValueError(f"Cannot handle non-class type {typ}.")
    elif issubclass(typ, Mapping):
        return parse_mapping(Any, Any, value)
    elif issubclass(typ, tuple | list):
        return parse_iterable(Any, value)
    else:
        if isinstance(value, typ):
            return value
        else:
            return typ(value)


def parse_iterable(value_type, values):
    _assert_type(values, tuple | list)
    if not isinstance(value_type, Iterable):
        return tuple(parse(value_type, val) for val in values)
    else:
        return tuple(
            parse(typ, val) for typ, val in zip(value_type, values, strict=True)
        )


def parse_data_class(clz, kwargs):
    if not is_dataclass(clz):
        raise ValueError(f"{clz} is not a dataclass")
    _assert_type(kwargs, dict)
    parsed_kwargs = {}
    for name, field in clz.__dataclass_fields__.items():
        if name in kwargs:
            value = parse(field.type, kwargs[name])
            parsed_kwargs[name] = value
    return clz(**parsed_kwargs)


def parse_mapping(key_type, value_type, value):
    _assert_type(value, Mapping)
    return {k: parse(value_type, v) for k, v in value.items()}


def parse_generic_alias(alias, value):
    _assert_type(alias, GenericAlias)
    origin = alias.__origin__
    # _assert_type(value, origin)  # pyyaml stores as list by default
    args = alias.__args__
    if issubclass(origin, tuple):
        if len(args) == 1 or args[1] == ...:
            return parse_iterable(args[0], value)
        else:
            return parse_iterable(args, value)
    elif issubclass(origin, Mapping):
        key_type, value_type = args
        return parse_mapping(key_type, value_type, value)
    elif issubclass(origin, Iterable):
        return parse_iterable(args[0], value)
    else:
        raise ValueError(f"Cannot parse generic alias {origin}")


def parse_union_type(union_type, value):
    # parse greedily
    for type in union_type.__args__:
        try:
            return parse(type, value)
        except:
            pass
    raise ValueError(f"Cannot process {union_type} for {value}")


def to_yaml(obj, stream=None):
    assert is_dataclass(type(obj))
    return yaml.dump(asdict(obj), stream, Dumper=NoAliasDumper)


def needs_flattening(obj):
    if isinstance(obj, Iterable) and not isinstance(obj, Mapping):
        for elem in obj:
            return needs_flattening(elem)
    else:
        return isinstance(obj, dict)


def flatten_dict(d, prefix=()):
    for key, val in d.items():
        prefix_ = prefix + (key,)
        if isinstance(val, dict):
            for k, v in flatten_dict(val, prefix=prefix_):
                yield (k, v)
        elif isinstance(val, list | tuple):
            if needs_flattening(val):
                for i, v in enumerate(val):
                    for k, v in flatten_dict(v, prefix_ + (str(i),)):
                        yield (k, v)
            else:
                yield (".".join(prefix_), val)
        else:
            yield (".".join(prefix_), val)


def to_hparam_dict(obj):
    assert is_dataclass(type(obj))
    return dict(flatten_dict(asdict(obj)))


def from_yaml(clz: Type, stream):
    return parse(clz, yaml.load(stream, yaml.Loader))
