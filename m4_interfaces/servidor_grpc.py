import grpc
from concurrent import futures
import convertidor_pb2
import convertidor_pb2_grpc

# Tipos de cambio (Obtenidos de Google)
TIPO_CAMBIO_USD = 0.058  # 1 MXN = 0.058 USD aprox
TIPO_CAMBIO_JPY = 9.20  # 1 MXN = 9. 20 JPY aprox

class ConvertidorServicer(convertidor_pb2_grpc.ConvertidorServicer):

    # request contiene los datos enviados por el cliente (el monto en MXN)
    def ConvertirADolar(self, request, context):
        convertido = request.monto_mxn * TIPO_CAMBIO_USD
        print(f"[Servidor] {request.monto_mxn} MXN -> {convertido:.2f} USD")
        # Construye y devuelve un objeto de respuesta al cliente
        return convertidor_pb2.ConversionResponse(
            monto_convertido=convertido,
            moneda_destino="USD",
            tipo_cambio=TIPO_CAMBIO_USD
        )

    def ConvertirAYen(self, request, context):
        convertido = request.monto_mxn * TIPO_CAMBIO_JPY
        print(f"[Servidor] {request.monto_mxn} MXN -> {convertido:.2f} JPY")
        # Construye y devuelve la respuesta al cliente.
        return convertidor_pb2.ConversionResponse(
            monto_convertido=convertido,
            moneda_destino="JPY",
            tipo_cambio=TIPO_CAMBIO_JPY
        )

# Función para configurar y arrancar el servidor
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Vincula la lógica de nuestra clase 'ConvertidorServicer' con el servidor
    convertidor_pb2_grpc.add_ConvertidorServicer_to_server(
        ConvertidorServicer(), server)
    # Abre el puerto 50051 para escuchar peticiones desde cualquier IP
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Servidor de conversión de divisas escuchando en el puerto 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()