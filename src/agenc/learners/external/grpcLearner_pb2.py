# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/grpcLearner.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17proto/grpcLearner.proto\"T\n\x0b\x44\x61taPackage\x12!\n\x08metadata\x18\x01 \x03(\x0b\x32\x0f.ColumnMetadata\x12\"\n\x0cobservations\x18\x02 \x03(\x0b\x32\x0c.Observation\"O\n\nPrediction\x12!\n\x0bpredictions\x18\x01 \x03(\x0b\x32\x0c.Observation\x12\x1e\n\x06status\x18\x02 \x01(\x0b\x32\x0e.StatusMessage\"`\n\x0e\x43olumnMetadata\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\"\n\x0c\x66\x65\x61ture_type\x18\x02 \x01(\x0e\x32\x0c.FeatureType\x12\x1c\n\tdata_type\x18\x03 \x01(\x0e\x32\t.DataType\"i\n\x0bObservation\x12!\n\x06\x66ields\x18\x01 \x03(\x0b\x32\x11.ObservationField\x12\'\n\x0btime_vector\x18\x02 \x01(\x0b\x32\r.VectorDoubleH\x00\x88\x01\x01\x42\x0e\n\x0c_time_vector\"\xd0\x01\n\x10ObservationField\x12\r\n\x03int\x18\x01 \x01(\x05H\x00\x12\x10\n\x06\x64ouble\x18\x02 \x01(\x01H\x00\x12 \n\nvector_int\x18\x03 \x01(\x0b\x32\n.VectorIntH\x00\x12&\n\rvector_double\x18\x04 \x01(\x0b\x32\r.VectorDoubleH\x00\x12 \n\nmatrix_int\x18\x05 \x01(\x0b\x32\n.MatrixIntH\x00\x12&\n\rmatrix_double\x18\x06 \x01(\x0b\x32\r.MatrixDoubleH\x00\x42\x07\n\x05\x66ield\"\x19\n\tVectorInt\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x05\"\x1c\n\x0cVectorDouble\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x01\"B\n\tMatrixInt\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x05\x12\x11\n\trow_count\x18\x02 \x01(\x05\x12\x14\n\x0c\x63olumn_count\x18\x03 \x01(\x05\"E\n\x0cMatrixDouble\x12\x0c\n\x04\x64\x61ta\x18\x01 \x03(\x01\x12\x11\n\trow_count\x18\x02 \x01(\x05\x12\x14\n\x0c\x63olumn_count\x18\x03 \x01(\x05\"h\n\rStatusMessage\x12\x17\n\x06status\x18\x01 \x01(\x0e\x32\x07.Status\x12\x1a\n\x08messages\x18\x02 \x03(\x0b\x32\x08.Message\x12\x15\n\x08progress\x18\x03 \x01(\x05H\x00\x88\x01\x01\x42\x0b\n\t_progress\"H\n\x07Message\x12\x1c\n\tlog_level\x18\x01 \x01(\x0e\x32\t.LogLevel\x12\x0e\n\x06sender\x18\x02 \x01(\t\x12\x0f\n\x07message\x18\x03 \x01(\t\"\x07\n\x05\x45mpty*W\n\x0b\x46\x65\x61tureType\x12\x19\n\x15\x46\x45\x41TURETYPE_UNDEFINED\x10\x00\x12\x15\n\x11\x46\x45\x41TURETYPE_INPUT\x10\x01\x12\x16\n\x12\x46\x45\x41TURETYPE_TARGET\x10\x02*\xa1\x01\n\x08\x44\x61taType\x12\x16\n\x12\x44\x41TATYPE_UNDEFINED\x10\x00\x12\x13\n\x0f\x44\x41TATYPE_SCALAR\x10\x01\x12\x13\n\x0f\x44\x41TATYPE_VECTOR\x10\x02\x12\x13\n\x0f\x44\x41TATYPE_MATRIX\x10\x03\x12\x1e\n\x1a\x44\x41TATYPE_SCALAR_TIMESERIES\x10\x04\x12\x1e\n\x1a\x44\x41TATYPE_VECTOR_TIMESERIES\x10\x05*Z\n\x06Status\x12\x14\n\x10STATUS_UNDEFINED\x10\x00\x12\x12\n\x0eSTATUS_RUNNING\x10\x01\x12\x13\n\x0fSTATUS_FINISHED\x10\x02\x12\x11\n\rSTATUS_FAILED\x10\x03*\x87\x01\n\x08LogLevel\x12\x16\n\x12LOGLEVEL_UNDEFINED\x10\x00\x12\x12\n\x0eLOGLEVEL_DEBUG\x10\x01\x12\x11\n\rLOGLEVEL_INFO\x10\x02\x12\x14\n\x10LOGLEVEL_WARNING\x10\x03\x12\x12\n\x0eLOGLEVEL_ERROR\x10\x04\x12\x12\n\x0eLOGLEVEL_FATAL\x10\x05\x32\x80\x01\n\x0f\x45xternalLearner\x12)\n\x05Train\x12\x0c.DataPackage\x1a\x0e.StatusMessage\"\x00\x30\x01\x12&\n\x07Predict\x12\x0c.DataPackage\x1a\x0b.Prediction\"\x00\x12\x1a\n\x06\x45xport\x12\x06.Empty\x1a\x06.Empty\"\x00\x42\x17\n\x15io.agenc.learner.grpcb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.grpcLearner_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\025io.agenc.learner.grpc'
  _FEATURETYPE._serialized_start=995
  _FEATURETYPE._serialized_end=1082
  _DATATYPE._serialized_start=1085
  _DATATYPE._serialized_end=1246
  _STATUS._serialized_start=1248
  _STATUS._serialized_end=1338
  _LOGLEVEL._serialized_start=1341
  _LOGLEVEL._serialized_end=1476
  _DATAPACKAGE._serialized_start=27
  _DATAPACKAGE._serialized_end=111
  _PREDICTION._serialized_start=113
  _PREDICTION._serialized_end=192
  _COLUMNMETADATA._serialized_start=194
  _COLUMNMETADATA._serialized_end=290
  _OBSERVATION._serialized_start=292
  _OBSERVATION._serialized_end=397
  _OBSERVATIONFIELD._serialized_start=400
  _OBSERVATIONFIELD._serialized_end=608
  _VECTORINT._serialized_start=610
  _VECTORINT._serialized_end=635
  _VECTORDOUBLE._serialized_start=637
  _VECTORDOUBLE._serialized_end=665
  _MATRIXINT._serialized_start=667
  _MATRIXINT._serialized_end=733
  _MATRIXDOUBLE._serialized_start=735
  _MATRIXDOUBLE._serialized_end=804
  _STATUSMESSAGE._serialized_start=806
  _STATUSMESSAGE._serialized_end=910
  _MESSAGE._serialized_start=912
  _MESSAGE._serialized_end=984
  _EMPTY._serialized_start=986
  _EMPTY._serialized_end=993
  _EXTERNALLEARNER._serialized_start=1479
  _EXTERNALLEARNER._serialized_end=1607
# @@protoc_insertion_point(module_scope)