import grpc
import convertidor_pb2
import convertidor_pb2_grpc

def run():
    # Abrir canal de comunicacion
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = convertidor_pb2_grpc.ConvertidorStub(channel)

        monto = float(input("Ingresa un monto en MXN: "))

        # Conversión a USD
        respuesta_usd = stub.ConvertirADolar(
            convertidor_pb2.ConversionRequest(monto_mxn=monto))
        print(f"{monto} MXN = {respuesta_usd.monto_convertido:.2f} "
              f"{respuesta_usd.moneda_destino} "
              f"(tasa: {respuesta_usd.tipo_cambio})")

        # Conversión a JPY
        respuesta_jpy = stub.ConvertirAYen(
            convertidor_pb2.ConversionRequest(monto_mxn=monto))
        print(f"{monto} MXN = {respuesta_jpy.monto_convertido:.2f} "
              f"{respuesta_jpy.moneda_destino} "
              f"(tasa: {respuesta_jpy.tipo_cambio})")

if __name__ == '__main__':
    run()