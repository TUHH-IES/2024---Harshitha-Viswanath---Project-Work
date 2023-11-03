# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import agenc.learners.external.grpcLearner_pb2 as proto_dot_grpcLearner__pb2


class ExternalLearnerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Train = channel.unary_stream(
                '/ExternalLearner/Train',
                request_serializer=proto_dot_grpcLearner__pb2.DataPackage.SerializeToString,
                response_deserializer=proto_dot_grpcLearner__pb2.StatusMessage.FromString,
                )
        self.Predict = channel.unary_unary(
                '/ExternalLearner/Predict',
                request_serializer=proto_dot_grpcLearner__pb2.DataPackage.SerializeToString,
                response_deserializer=proto_dot_grpcLearner__pb2.Prediction.FromString,
                )
        self.Export = channel.unary_unary(
                '/ExternalLearner/Export',
                request_serializer=proto_dot_grpcLearner__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_grpcLearner__pb2.Empty.FromString,
                )


class ExternalLearnerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Train(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Export(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ExternalLearnerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Train': grpc.unary_stream_rpc_method_handler(
                    servicer.Train,
                    request_deserializer=proto_dot_grpcLearner__pb2.DataPackage.FromString,
                    response_serializer=proto_dot_grpcLearner__pb2.StatusMessage.SerializeToString,
            ),
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=proto_dot_grpcLearner__pb2.DataPackage.FromString,
                    response_serializer=proto_dot_grpcLearner__pb2.Prediction.SerializeToString,
            ),
            'Export': grpc.unary_unary_rpc_method_handler(
                    servicer.Export,
                    request_deserializer=proto_dot_grpcLearner__pb2.Empty.FromString,
                    response_serializer=proto_dot_grpcLearner__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ExternalLearner', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ExternalLearner(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Train(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/ExternalLearner/Train',
            proto_dot_grpcLearner__pb2.DataPackage.SerializeToString,
            proto_dot_grpcLearner__pb2.StatusMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ExternalLearner/Predict',
            proto_dot_grpcLearner__pb2.DataPackage.SerializeToString,
            proto_dot_grpcLearner__pb2.Prediction.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Export(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ExternalLearner/Export',
            proto_dot_grpcLearner__pb2.Empty.SerializeToString,
            proto_dot_grpcLearner__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)