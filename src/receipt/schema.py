"""版本化业务 schema；FATURA 字段必须由训练侧显式注册。"""

from copy import deepcopy
import math

from jsonschema import Draft202012Validator

RECEIPT_FIELDS = (
    "merchant_name", "address", "phone", "date", "time", "order_number",
    "items", "subtotal", "tax", "total", "payment_method",
)
KEY_FIELDS = ("merchant_name", "date", "total")
FIELD_STATUSES = frozenset({
    "present_readable", "not_printed", "present_unreadable", "unannotated",
})

_TEXT = {"type": ["string", "null"]}
_RECEIPT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        field: deepcopy(_TEXT) for field in RECEIPT_FIELDS
    },
    "required": list(RECEIPT_FIELDS),
    "additionalProperties": False,
}
_RECEIPT_SCHEMA["properties"]["items"] = {
    "type": ["array", "null"],
    "items": {
        "type": "object",
        "properties": {field: deepcopy(_TEXT) for field in ("name", "quantity", "price")},
        "required": ["name", "quantity", "price"],
        "additionalProperties": False,
    },
}
_JSON_VALUE = {
    "anyOf": [
        {"type": ["string", "number", "null"]},
        {"type": "array", "items": {"$ref": "#/$defs/value"}},
        {"type": "object", "additionalProperties": {"$ref": "#/$defs/value"}},
    ]
}
_GENERIC_INVOICE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "$defs": {"value": _JSON_VALUE},
    "additionalProperties": {"$ref": "#/$defs/value"},
}
_SCHEMAS = {"receipt_v1": _RECEIPT_SCHEMA, "invoice_json_v1": _GENERIC_INVOICE}


def register_schema(schema_version, schema):
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("schema_version 必须是非空字符串")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("注册 schema 必须是 JSON 对象 schema")
    frozen = deepcopy(schema)
    if schema_version != "invoice_json_v1":
        if not isinstance(frozen.get("properties"), dict) or not frozen["properties"]:
            raise ValueError("固定 schema 必须显式提供训练侧 properties")
        frozen["additionalProperties"] = False
    Draft202012Validator.check_schema(frozen)
    if schema_version in _SCHEMAS and _SCHEMAS[schema_version] != frozen:
        raise ValueError("已冻结的 schema 版本不可改写，请注册新版本")
    _SCHEMAS[schema_version] = frozen


def register_invoice_fatura_schema(schema):
    register_schema("invoice_fatura_v1", schema)


register_fatura_schema = register_invoice_fatura_schema


def get_schema(schema_version="receipt_v1", fields=None):
    if schema_version not in _SCHEMAS:
        raise ValueError(f"schema 未注册: {schema_version}；FATURA 必须使用训练数据固定 schema")
    schema = deepcopy(_SCHEMAS[schema_version])
    if fields is None:
        return schema
    if isinstance(fields, (str, bytes)):
        raise ValueError("fields 必须是字段列表，不能是字符串")
    fields = list(fields)
    if any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("字段名必须是非空字符串")
    if len(set(fields)) != len(fields):
        raise ValueError("fields 不能包含重复字段")
    properties = schema.get("properties", {})
    if schema_version == "invoice_json_v1":
        schema["properties"] = {field: {"$ref": "#/$defs/value"} for field in fields}
    else:
        unknown = set(fields) - properties.keys()
        if unknown:
            raise ValueError(f"未注册字段: {', '.join(sorted(unknown))}")
        schema["properties"] = {field: properties[field] for field in fields}
    schema["required"] = fields
    schema["additionalProperties"] = False
    return schema


def json_value_issues(value, path="$", active=None):
    active = set() if active is None else active
    if value is None or isinstance(value, (str, bool, int)):
        return []
    if isinstance(value, float):
        return [] if math.isfinite(value) else [f"{path}: non_finite_number"]
    if not isinstance(value, (dict, list)):
        return [f"{path}: non_json_type"]
    if id(value) in active:
        return [f"{path}: circular_reference"]
    active.add(id(value))
    issues = []
    entries = value.items() if isinstance(value, dict) else enumerate(value)
    for key, child in entries:
        if isinstance(value, dict) and not isinstance(key, str):
            issues.append(f"{path}: non_string_key")
        issues.extend(json_value_issues(child, f"{path}/{key}", active))
    active.remove(id(value))
    return issues


def validate_output(data, schema_version="receipt_v1", fields=None):
    schema = get_schema(schema_version, fields)
    issues = json_value_issues(data)
    if issues:
        return issues
    try:
        errors = sorted(Draft202012Validator(schema).iter_errors(data),
                        key=lambda error: (str(list(error.absolute_path)), error.message))
        for error in errors:
            path = "/".join(str(part) for part in error.absolute_path)
            issues.append(f"$/{path}: {error.message}")
    except RecursionError:
        issues.append("$: nesting_too_deep")
    return issues
