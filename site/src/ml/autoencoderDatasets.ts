// Tiny grayscale "images" for the autoencoder playground. Each scenario makes images that live in
// W×W pixels but really vary in only one or two ways — a low-dimensional manifold hidden in a
// high-dimensional pixel space. That gap is the whole point: the autoencoder has to discover the few
// numbers that matter. Pixel values are in [0, 1] to match the network's sigmoid output.

export interface AutoencoderScenario {
    id: string;
    label: string;
    blurb: string;
    width: number;        // images are width × width
    intrinsicDim: number; // how many numbers really vary (what a good code size should match)
    /** Generate `count` images plus a scalar "colour" per image (its hidden factor), for the latent map. */
    generate: (seed: number, count: number) => { images: number[][]; colors: number[] };
    /** A clean, centred example used for the denoising demo. */
    example: () => number[];
}

function blobImage(width: number, cx: number, cy: number, sigma: number): number[] {
    const img: number[] = [];
    for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) img.push(Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / sigma));
    return img;
}

function barImage(width: number, col: number): number[] {
    const img: number[] = [];
    for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) img.push(Math.exp(-((x - col) ** 2) / 1.5));
    return img;
}

export const AUTOENCODER_SCENARIOS: AutoencoderScenario[] = [
    {
        id: 'crowd-blobs',
        label: 'Crowd blobs',
        blurb: 'Each image is a soft blob — think of where the crowd clustered on the café floor that day. It moves in two ways (across and up), so a 2-number code should capture it exactly.',
        width: 8,
        intrinsicDim: 2,
        generate: (seed, count) => {
            let s = seed;
            const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
            const images: number[][] = [];
            const colors: number[] = [];
            for (let i = 0; i < count; i++) {
                const cx = 1.5 + rand() * 5;
                const cy = 1.5 + rand() * 5;
                images.push(blobImage(8, cx, cy, 5));
                colors.push(cx / 8); // colour by horizontal position
            }
            return { images, colors };
        },
        example: () => blobImage(8, 4, 4, 5),
    },
    {
        id: 'sliding-bar',
        label: 'Sliding bar',
        blurb: 'A single vertical stripe that only slides left and right — one hidden number. The latent codes line up into a clear gradient, the autoencoder surfacing that the data has just one real degree of freedom.',
        width: 8,
        intrinsicDim: 1,
        generate: (seed, count) => {
            let s = seed;
            const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
            const images: number[][] = [];
            const colors: number[] = [];
            for (let i = 0; i < count; i++) {
                const col = 0.8 + rand() * 6.4;
                images.push(barImage(8, col));
                colors.push(col / 8);
            }
            return { images, colors };
        },
        example: () => barImage(8, 4),
    },
];
