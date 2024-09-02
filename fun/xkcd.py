#!/usr/bin/env python3

from matplotlib import pyplot as plt


def xkcd_imagenet():
    with plt.xkcd(scale=2, length=200):

        model_data = [
            ('AlexNet', 2012, 74.6, 'w/o BN'),
            ('VGG', 2014, 91.516, 'w/o BN'),
            ('BN-Inception', 2015, 93.5, 'w/ BN'),
            ('ResNet', 2015, 95.434, 'w/ BN'),
            ('ResNeXt', 2016, 96.196, 'w/ BN'),
            ('SENet', 2017, 96.878, 'w/ BN'),
        ]

        with_bn = [data for *data, bn_type in model_data if bn_type == 'w/ BN']
        without_bn = [data for *data, bn_type in model_data if bn_type == 'w/o BN']

        with_bn_models, with_bn_years, with_bn_vals = zip(*with_bn)
        without_bn_models, without_bn_years, without_bn_vals = zip(*without_bn)

        plt.scatter(with_bn_years, with_bn_vals, color='blue', label='w/ BN', s=100)
        plt.scatter(without_bn_years, without_bn_vals, color='red', label='w/o BN', s=100)

        plt.xlabel('Year')
        plt.ylabel('Top-5 Acc (%)')
        plt.title('Top-5 Acc of CNN Models Over Years')

        plt.axvline(x=2014, color='gray')
        plt.axhline(y=92, color='gray')

        for model, year, val in with_bn + without_bn:
            offset = 0.1
            plt.text(year + offset, val + offset, model, fontsize=10, ha='left', va='center')

        plt.legend()

        plt.savefig("xkcd.png")
        plt.show()


if __name__ == "__main__":
    xkcd_imagenet()
