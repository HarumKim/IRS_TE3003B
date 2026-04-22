import rclpy
from rclpy.node import Node
from currency_interfaces.srv import ConvertCurrency


class ClienteConvertidor(Node):

    def __init__(self):
        super().__init__('cliente_convertidor')
        # Equivalente al canal + stub de gRPC
        self.client = self.create_client(ConvertCurrency, 'convertir_divisa')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando al servidor de divisas...')

    def convertir(self, monto, moneda_destino):
        request = ConvertCurrency.Request()
        request.amount = monto
        request.from_currency = 'MXN'
        request.to_currency = moneda_destino

        # Llamada asíncrona al servicio 
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main(args=None):
    rclpy.init(args=args)
    node = ClienteConvertidor()

    monto = float(input('Ingresa un monto en MXN: '))

    # Conversión a USD
    respuesta_usd = node.convertir(monto, 'USD')
    print(f'{monto} MXN = {respuesta_usd.converted_amount:.2f} '
          f'{respuesta_usd.target_currency} '
          f'(tasa: {respuesta_usd.exchange_rate})')

    # Conversión a JPY
    respuesta_jpy = node.convertir(monto, 'JPY')
    print(f'{monto} MXN = {respuesta_jpy.converted_amount:.2f} '
          f'{respuesta_jpy.target_currency} '
          f'(tasa: {respuesta_jpy.exchange_rate})')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
