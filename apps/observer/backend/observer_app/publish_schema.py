"""JSON Schema (draft 2020-12 subset) for the published files: inferred from real builder output, and a small
validator for the same subset (the observer runtime has no ``jsonschema`` package and needs none).

``infer(samples)`` describes exactly what the given bodies contain: the JSON types seen at each place, the keys
present in every sample (``required``), array items, and id-keyed maps (``additionalProperties``). It states
nothing about values it did not see, so a schema inferred from few bodies is narrower than the builders' full
range: a field seen only as a string may be null in another game. Schemas never set ``additionalProperties:
false`` for a keyed object, so adding a field stays a compatible (v1) change.

Keywords used: ``type`` (name or list), ``properties``, ``required``, ``items``, ``additionalProperties`` (a
schema), ``const``, ``enum``, ``minimum``. ``validate`` checks those and nothing else.
"""
from __future__ import annotations

import re

DRAFT = 'https://json-schema.org/draft/2020-12/schema'
_MAP_KEY = re.compile(r'\d+(?::\d+)*')  # player ids, pitch keys: a map keyed by an identifier, not a record


def _type(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, int):
        return 'integer'
    if isinstance(value, float):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    raise TypeError(f'not a JSON value: {type(value).__name__}')


def infer(samples):
    """The narrowest schema (of the subset above) every value in ``samples`` satisfies. No sample: ``{}``."""
    samples = list(samples)
    if not samples:
        return {}
    types = {_type(value) for value in samples}
    if 'number' in types:
        types.discard('integer')  # JSON Schema: every integer is a number
    order = ('object', 'array', 'string', 'number', 'integer', 'boolean', 'null')
    names = [name for name in order if name in types]
    schema = {'type': names[0] if len(names) == 1 else names}
    objects = [value for value in samples if isinstance(value, dict)]
    arrays = [value for value in samples if isinstance(value, list)]
    if objects:
        keys = sorted({key for value in objects for key in value})
        if keys and all(_MAP_KEY.fullmatch(key) for key in keys):
            schema['additionalProperties'] = infer(item for value in objects for item in value.values())
        elif keys:
            schema['properties'] = {key: infer(value[key] for value in objects if key in value) for key in keys}
            required = [key for key in keys if all(key in value for value in objects)]
            if required:
                schema['required'] = required
    if arrays:
        items = [item for value in arrays for item in value]
        if items:
            schema['items'] = infer(items)
    return schema


def validate(instance, schema, path='$'):
    """Every way ``instance`` breaks ``schema``, as ``'<json path>: <what>'`` strings (empty = valid)."""
    errors = []
    if 'const' in schema and instance != schema['const']:
        errors.append(f"{path}: expected {schema['const']!r}, got {instance!r}")
    if 'enum' in schema and instance not in schema['enum']:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']!r}")
    if 'type' in schema:
        allowed = [schema['type']] if isinstance(schema['type'], str) else schema['type']
        actual = _type(instance)
        if actual not in allowed and not (actual == 'integer' and 'number' in allowed):
            return errors+[f'{path}: {actual} is not {" or ".join(allowed)}']
    if isinstance(instance, (int, float)) and not isinstance(instance, bool) and 'minimum' in schema and instance < schema['minimum']:
        errors.append(f"{path}: {instance} is below {schema['minimum']}")
    if isinstance(instance, dict):
        errors += [f'{path}: missing {key!r}' for key in schema.get('required', []) if key not in instance]
        known = schema.get('properties', {})
        for key, value in instance.items():
            if key in known:
                errors += validate(value, known[key], f'{path}.{key}')
            elif isinstance(schema.get('additionalProperties'), dict):
                errors += validate(value, schema['additionalProperties'], f'{path}.{key}')
    if isinstance(instance, list) and 'items' in schema:
        for index, value in enumerate(instance):
            errors += validate(value, schema['items'], f'{path}[{index}]')
    return errors


def problems(instance, schema, pointer=''):
    """Like ``validate`` but for reports on real data: each problem is ``{'pointer', 'keyword', 'seen'}`` where
    ``pointer`` is a JSON pointer made of the schema's own property names, array indices and ``*`` for any key
    the schema does not name; ``seen`` is the JSON TYPE found there (``missing`` for an absent required key).
    Nothing from the instance's values or its own keys is reported."""
    out = []
    add = lambda keyword, seen: out.append({'pointer': pointer or '/', 'keyword': keyword, 'seen': seen})
    if 'const' in schema and instance != schema['const']:
        add('const', _type(instance))
    if 'enum' in schema and instance not in schema['enum']:
        add('enum', _type(instance))
    if 'type' in schema:
        allowed = [schema['type']] if isinstance(schema['type'], str) else schema['type']
        actual = _type(instance)
        if actual not in allowed and not (actual == 'integer' and 'number' in allowed):
            add('type', actual)
            return out
    if isinstance(instance, (int, float)) and not isinstance(instance, bool) and 'minimum' in schema and instance < schema['minimum']:
        add('minimum', _type(instance))
    if isinstance(instance, dict):
        for key in schema.get('required', []):
            if key not in instance:
                out.append({'pointer': f'{pointer}/{key}', 'keyword': 'required', 'seen': 'missing'})
        known = schema.get('properties', {})
        for key, value in instance.items():
            token = str(key).replace('~', '~0').replace('/', '~1')
            if key in known:
                out += problems(value, known[key], f'{pointer}/{token}')
            elif isinstance(schema.get('additionalProperties'), dict):
                # a key the schema does not name is an identifier from the data (a player id...): never reported
                out += problems(value, schema['additionalProperties'], f'{pointer}/*')
    if isinstance(instance, list) and 'items' in schema:
        for index, value in enumerate(instance):
            out += problems(value, schema['items'], f'{pointer}/{index}')
    return out


def publish_block(*, final):
    """The one key the publisher adds to every file. ``final`` True: the file kind carries the final mark."""
    properties = {'contract': {'const': 'v1'}, 'sequence': {'type': 'integer', 'minimum': 1},
                  'published_at': {'type': 'string'}}
    if final:
        properties['final'] = {'type': 'boolean'}
    return {'type': 'object', 'properties': properties, 'required': sorted(properties)}


def document(kind, samples, *, final, source):
    """The schema file of one published file kind: ``infer`` over the bodies (without ``publish``), plus the
    hand-written ``publish`` block and the ``schema`` name those bodies carry."""
    bodies = [{key: value for key, value in sample.items() if key != 'publish'} for sample in samples]
    schema = infer(bodies)
    names = sorted({body['schema'] for body in bodies if isinstance(body.get('schema'), str)})
    if names:
        schema['properties']['schema'] = {'enum': names}
    schema['properties']['publish'] = publish_block(final=final)
    schema['required'] = sorted({*schema.get('required', []), 'publish'})
    return {'$schema': DRAFT, '$id': f'pitcheezy-publish-v1-{kind}', 'title': f'Pitcheezy public publish v1: {kind}',
            'description': (f'Inferred from {len(bodies)} synthetic fixture bodies ({source}) built by the same code that '
                            'answers the service routes; the publish block is fixed. It describes what those bodies '
                            'contain, not every value a real game can produce: treat a type listed here as the minimum '
                            'a reader must accept, and accept unknown keys.')} | schema
