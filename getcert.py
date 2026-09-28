import ssl
import socket
from cryptography import x509
from cryptography.hazmat.primitives import serialization


def extract_full_chain(hostname, port=443, output_file="full_chain.pem"):
    # We use CERT_NONE to establish the connection regardless of trust
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    try:
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                # Retrieve the certificates sent by the server in binary DER format
                # This returns a list of certificates in the chain
                chain_der = ssock.get_verified_chain() if hasattr(ssock, 'get_verified_chain') else None

                # Fallback for Python versions < 3.13
                if chain_der is None:
                    # getpeercert(binary_form=True) only returns the leaf.
                    # To get the chain in older Python versions, we must use the
                    # socket's raw handshake or external tools.
                    # However, we can manually attempt to fetch the issuer if provided.
                    print("Your Python version doesn't support get_verified_chain().")
                    print("Attempting to extract the leaf; you may need to manually add the Sub-CA.")
                    cert_der = [ssock.getpeercert(binary_form=True)]
                else:
                    cert_der = chain_der

                with open(output_file, "w") as f:
                    for der_cert in cert_der:
                        # Convert DER to X509 object for processing
                        cert = x509.load_der_x509_certificate(der_cert)
                        # Export to PEM
                        pem = cert.public_bytes(serialization.Encoding.PEM)
                        f.write(pem.decode('utf-8'))

                print(f"Successfully wrote {len(cert_der)} certificates to {output_file}")

    except Exception as e:
        print(f"Error: {e}")


extract_full_chain("pypi.org")
