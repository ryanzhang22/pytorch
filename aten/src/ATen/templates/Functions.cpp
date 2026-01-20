#include <array>

#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <c10/core/Allocator.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

namespace at {

static void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}

static std::optional<Tensor> maybe_create_proxy(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  if (out.strides() != strides) {
    return at::detail::empty_strided_cpu(sizes, strides, options);
  }
  return std::nullopt;
}

struct structured_mul_out_out final : public at::native::structured_mul_out {
    structured_mul_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}
    void set_output_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
        if (C10_UNLIKELY(maybe_proxy.has_value())) {
            proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
        }
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    void set_output_raw_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        const auto& out = outputs_[output_idx].get();
        resize_out(out, sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_mul_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get();
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
    std::array<::std::optional<Tensor>, 1> proxy_outputs_;
};

Tensor TensorMaker::make_tensor() {
   AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
   tracer::impl::NoTracerDispatchMode tracer_guard{};

   check_size_nonnegative(sizes_);

   TORCH_CHECK_VALUE(
       !deleter_ || !ctx_,
       "The deleter and context arguments are mutually exclusive.");

   if (device_ == std::nullopt) {
     device_ = globalContext().getDeviceFromPtr(data_, opts_.device().type());
   }

   if (opts_.device().has_index()) {
     // clang-format off
     TORCH_CHECK_VALUE(
         opts_.device() == *device_,
         "Specified device ", opts_.device(), " does not match device of data ", *device_);
     // clang-format on
   }

   std::size_t size_bytes = computeStorageSize();

   DataPtr data_ptr{};
   if (deleter_) {
     data_ptr = makeDataPtrFromDeleter();
   } else {
     data_ptr = makeDataPtrFromContext();
   }

   TORCH_CHECK(!resizeable_ || allocator_ != nullptr, "Must specify an allocator with allocator() if you want to use resizeable_storage()");
   Storage storage{Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr), /*allocator=*/allocator_, /*resizable=*/resizeable_};

   Tensor tensor = detail::make_tensor<TensorImpl>(
       std::move(storage), opts_.computeDispatchKey(), opts_.dtype());

  TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  if (strides_) {
    tensor_impl->set_sizes_and_strides(sizes_, *strides_);
  } else {
    tensor_impl->set_sizes_contiguous(sizes_);
  }
  if (storage_offset_) {
    tensor_impl->set_storage_offset(*storage_offset_);
  }

  tensor_impl->set_requires_grad(opts_.requires_grad());

  return tensor;
 }

 std::size_t TensorMaker::computeStorageSize() const noexcept {
   std::size_t itemsize = opts_.dtype().itemsize();

   if (strides_) {
     auto storage_size = detail::computeStorageNbytes(sizes_, *strides_, itemsize);
     if (storage_offset_) {
       storage_size += storage_offset_.value() * itemsize;
     }
     return storage_size;
   }

   std::size_t size = 1;
   for (std::int64_t s : sizes_) {
     size *= static_cast<std::size_t>(s);
   }
   auto storage_size = size * itemsize;
   if (storage_offset_) {
     storage_size += storage_offset_.value() * itemsize;
   }
   return storage_size;
 }

 inline DataPtr TensorMaker::makeDataPtrFromDeleter() noexcept {
   return InefficientStdFunctionContext::makeDataPtr(data_, std::move(deleter_), *device_);
 }

 inline DataPtr TensorMaker::makeDataPtrFromContext() noexcept {
   return DataPtr{data_, ctx_.release(), ctx_.get_deleter(), *device_};
 }

 IntArrayRef TensorMaker::makeTempSizes() const noexcept {
   static std::int64_t zeros[5] = {0, 0, 0, 0, 0};
   if (opts_.has_memory_format()) {
     MemoryFormat format = *opts_.memory_format_opt();
     if (format == MemoryFormat::ChannelsLast) {
       return IntArrayRef(zeros, 4);
     }
     if (format == MemoryFormat::ChannelsLast3d) {
       return IntArrayRef(zeros, 5);
     }
   }
   return IntArrayRef(zeros, 1);
 }

Tensor mul(const Tensor& self, const Tensor& other) {
  return self.mul(other);
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return self.mul_(other);
}

Tensor& mul_out(Tensor& out, const Tensor& self, const Tensor& other) {
  structured_mul_out_out op(out);
  op.meta(self, other);
  op.impl(self, other, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value()) op.outputs_[0].get().copy_(*op.proxy_outputs_[0]);
  return out;
}

Tensor& mul_outf(const Tensor& self, const Tensor& other, Tensor& out) {
  return at::mul_out(out, self, other);
}

} // namespace at
