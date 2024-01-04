"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _Status:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _StatusEnumTypeWrapper(
    google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[
        _Status.ValueType
    ],
    builtins.type,
):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    STATUS_UNDEFINED: _Status.ValueType  # 0
    STATUS_RUNNING: _Status.ValueType  # 1
    STATUS_FINISHED: _Status.ValueType  # 2
    STATUS_FAILED: _Status.ValueType  # 3

class Status(_Status, metaclass=_StatusEnumTypeWrapper): ...

STATUS_UNDEFINED: Status.ValueType  # 0
STATUS_RUNNING: Status.ValueType  # 1
STATUS_FINISHED: Status.ValueType  # 2
STATUS_FAILED: Status.ValueType  # 3
global___Status = Status

class _LogLevel:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _LogLevelEnumTypeWrapper(
    google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[
        _LogLevel.ValueType
    ],
    builtins.type,
):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    LOGLEVEL_UNDEFINED: _LogLevel.ValueType  # 0
    LOGLEVEL_DEBUG: _LogLevel.ValueType  # 1
    LOGLEVEL_INFO: _LogLevel.ValueType  # 2
    LOGLEVEL_WARNING: _LogLevel.ValueType  # 3
    LOGLEVEL_ERROR: _LogLevel.ValueType  # 4
    LOGLEVEL_FATAL: _LogLevel.ValueType  # 5

class LogLevel(_LogLevel, metaclass=_LogLevelEnumTypeWrapper): ...

LOGLEVEL_UNDEFINED: LogLevel.ValueType  # 0
LOGLEVEL_DEBUG: LogLevel.ValueType  # 1
LOGLEVEL_INFO: LogLevel.ValueType  # 2
LOGLEVEL_WARNING: LogLevel.ValueType  # 3
LOGLEVEL_ERROR: LogLevel.ValueType  # 4
LOGLEVEL_FATAL: LogLevel.ValueType  # 5
global___LogLevel = LogLevel

@typing_extensions.final
class DataPackage(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INPUTS_FIELD_NUMBER: builtins.int
    OUTPUTS_FIELD_NUMBER: builtins.int
    @property
    def inputs(
        self,
    ) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[
        global___DataRow
    ]: ...
    @property
    def outputs(
        self,
    ) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[
        global___DataRow
    ]: ...
    def __init__(
        self,
        *,
        inputs: collections.abc.Iterable[global___DataRow] | None = ...,
        outputs: collections.abc.Iterable[global___DataRow] | None = ...,
    ) -> None: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "inputs", b"inputs", "outputs", b"outputs"
        ],
    ) -> None: ...

global___DataPackage = DataPackage

@typing_extensions.final
class Prediction(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PREDICTIONS_FIELD_NUMBER: builtins.int
    STATUS_FIELD_NUMBER: builtins.int
    @property
    def predictions(
        self,
    ) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[
        global___DataRow
    ]: ...
    @property
    def status(self) -> global___StatusMessage: ...
    def __init__(
        self,
        *,
        predictions: collections.abc.Iterable[global___DataRow] | None = ...,
        status: global___StatusMessage | None = ...,
    ) -> None: ...
    def HasField(
        self, field_name: typing_extensions.Literal["status", b"status"]
    ) -> builtins.bool: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "predictions", b"predictions", "status", b"status"
        ],
    ) -> None: ...

global___Prediction = Prediction

@typing_extensions.final
class DataRow(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FIELDS_FIELD_NUMBER: builtins.int
    @property
    def fields(
        self,
    ) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[
        global___DataField
    ]: ...
    def __init__(
        self,
        *,
        fields: collections.abc.Iterable[global___DataField] | None = ...,
    ) -> None: ...
    def ClearField(
        self, field_name: typing_extensions.Literal["fields", b"fields"]
    ) -> None: ...

global___DataRow = DataRow

@typing_extensions.final
class DataField(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INT_FIELD_NUMBER: builtins.int
    DOUBLE_FIELD_NUMBER: builtins.int
    VECTOR_INT_FIELD_NUMBER: builtins.int
    VECTOR_DOUBLE_FIELD_NUMBER: builtins.int
    MATRIX_INT_FIELD_NUMBER: builtins.int
    MATRIX_DOUBLE_FIELD_NUMBER: builtins.int
    int: builtins.int
    double: builtins.float
    @property
    def vector_int(self) -> global___VectorInt: ...
    @property
    def vector_double(self) -> global___VectorDouble: ...
    @property
    def matrix_int(self) -> global___MatrixInt: ...
    @property
    def matrix_double(self) -> global___MatrixDouble: ...
    def __init__(
        self,
        *,
        int: builtins.int = ...,
        double: builtins.float = ...,
        vector_int: global___VectorInt | None = ...,
        vector_double: global___VectorDouble | None = ...,
        matrix_int: global___MatrixInt | None = ...,
        matrix_double: global___MatrixDouble | None = ...,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing_extensions.Literal[
            "double",
            b"double",
            "field",
            b"field",
            "int",
            b"int",
            "matrix_double",
            b"matrix_double",
            "matrix_int",
            b"matrix_int",
            "vector_double",
            b"vector_double",
            "vector_int",
            b"vector_int",
        ],
    ) -> builtins.bool: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "double",
            b"double",
            "field",
            b"field",
            "int",
            b"int",
            "matrix_double",
            b"matrix_double",
            "matrix_int",
            b"matrix_int",
            "vector_double",
            b"vector_double",
            "vector_int",
            b"vector_int",
        ],
    ) -> None: ...
    def WhichOneof(
        self, oneof_group: typing_extensions.Literal["field", b"field"]
    ) -> (
        typing_extensions.Literal[
            "int",
            "double",
            "vector_int",
            "vector_double",
            "matrix_int",
            "matrix_double",
        ]
        | None
    ): ...

global___DataField = DataField

@typing_extensions.final
class VectorInt(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATA_FIELD_NUMBER: builtins.int
    @property
    def data(
        self,
    ) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[
        builtins.int
    ]: ...
    def __init__(
        self,
        *,
        data: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(
        self, field_name: typing_extensions.Literal["data", b"data"]
    ) -> None: ...

global___VectorInt = VectorInt

@typing_extensions.final
class VectorDouble(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATA_FIELD_NUMBER: builtins.int
    @property
    def data(
        self,
    ) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[
        builtins.float
    ]: ...
    def __init__(
        self,
        *,
        data: collections.abc.Iterable[builtins.float] | None = ...,
    ) -> None: ...
    def ClearField(
        self, field_name: typing_extensions.Literal["data", b"data"]
    ) -> None: ...

global___VectorDouble = VectorDouble

@typing_extensions.final
class MatrixInt(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATA_FIELD_NUMBER: builtins.int
    ROW_COUNT_FIELD_NUMBER: builtins.int
    COLUMN_COUNT_FIELD_NUMBER: builtins.int
    @property
    def data(
        self,
    ) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[
        builtins.int
    ]: ...
    row_count: builtins.int
    column_count: builtins.int
    def __init__(
        self,
        *,
        data: collections.abc.Iterable[builtins.int] | None = ...,
        row_count: builtins.int = ...,
        column_count: builtins.int = ...,
    ) -> None: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "column_count",
            b"column_count",
            "data",
            b"data",
            "row_count",
            b"row_count",
        ],
    ) -> None: ...

global___MatrixInt = MatrixInt

@typing_extensions.final
class MatrixDouble(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATA_FIELD_NUMBER: builtins.int
    ROW_COUNT_FIELD_NUMBER: builtins.int
    COLUMN_COUNT_FIELD_NUMBER: builtins.int
    @property
    def data(
        self,
    ) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[
        builtins.float
    ]: ...
    row_count: builtins.int
    column_count: builtins.int
    def __init__(
        self,
        *,
        data: collections.abc.Iterable[builtins.float] | None = ...,
        row_count: builtins.int = ...,
        column_count: builtins.int = ...,
    ) -> None: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "column_count",
            b"column_count",
            "data",
            b"data",
            "row_count",
            b"row_count",
        ],
    ) -> None: ...

global___MatrixDouble = MatrixDouble

@typing_extensions.final
class StatusMessage(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    MESSAGES_FIELD_NUMBER: builtins.int
    PROGRESS_FIELD_NUMBER: builtins.int
    status: global___Status.ValueType
    @property
    def messages(
        self,
    ) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[
        global___Message
    ]: ...
    progress: builtins.int
    def __init__(
        self,
        *,
        status: global___Status.ValueType = ...,
        messages: collections.abc.Iterable[global___Message] | None = ...,
        progress: builtins.int | None = ...,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing_extensions.Literal[
            "_progress", b"_progress", "progress", b"progress"
        ],
    ) -> builtins.bool: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "_progress",
            b"_progress",
            "messages",
            b"messages",
            "progress",
            b"progress",
            "status",
            b"status",
        ],
    ) -> None: ...
    def WhichOneof(
        self, oneof_group: typing_extensions.Literal["_progress", b"_progress"]
    ) -> typing_extensions.Literal["progress"] | None: ...

global___StatusMessage = StatusMessage

@typing_extensions.final
class Message(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LOG_LEVEL_FIELD_NUMBER: builtins.int
    SENDER_FIELD_NUMBER: builtins.int
    MESSAGE_FIELD_NUMBER: builtins.int
    log_level: global___LogLevel.ValueType
    sender: builtins.str
    message: builtins.str
    def __init__(
        self,
        *,
        log_level: global___LogLevel.ValueType = ...,
        sender: builtins.str = ...,
        message: builtins.str = ...,
    ) -> None: ...
    def ClearField(
        self,
        field_name: typing_extensions.Literal[
            "log_level",
            b"log_level",
            "message",
            b"message",
            "sender",
            b"sender",
        ],
    ) -> None: ...

global___Message = Message

@typing_extensions.final
class Empty(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___Empty = Empty
