import rclpy
from rclpy.node import Node
from currency_interfaces.srv import ConvertCurrency

# Tipos de cambio (obtenidos de Google)
TIPO_CAMBIO_USD = 0.058  # 1 MXN = 0.058 USD aprox
TIPO_CAMBIO_JPY = 9.20   # 1 MXN = 9.20 JPY aprox

TASAS = {
    'USD': TIPO_CAMBIO_USD,
    'JPY': TIPO_CAMBIO_JPY,
}


class ConvertidorServicer(Node):

    def __init__(self):
        super().__init__('servidor_convertidor')
        self.srv = self.create_service(
            ConvertCurrency,
            'convertir_divisa',
            self.convertir_callback
        )
        self.get_logger().info('Servidor de conversión de divisas escuchando...')

    def convertir_callback(self, request, response):
        origen = request.from_currency.upper()
        destino = request.to_currency.upper()

        if origen != 'MXN' or destino not in TASAS:
            self.get_logger().warn(
                f'Par de divisas no soportado: {origen} -> {destino}'
            )
            response.converted_amount = 0.0
            response.target_currency = destino
            response.exchange_rate = 0.0
            return response

        tasa = TASAS[destino]
        response.converted_amount = request.amount * tasa
        response.target_currency = destino
        response.exchange_rate = tasa

        self.get_logger().info(
            f'[Servidor] {request.amount} {origen} -> '
            f'{response.converted_amount:.2f} {destino}'
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ConvertidorServicer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
